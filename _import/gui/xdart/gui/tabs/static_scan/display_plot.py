# -*- coding: utf-8 -*-
"""Plot rendering, waterfall, slice overlay, and mouse-tracking methods
for displayFrameWidget.

This mixin extracts ~600 lines of 1D-plot and waterfall logic from the
monolithic displayFrameWidget class.

The mixin is designed to be inherited by displayFrameWidget alongside
QWidget and DisplayDataMixin, so all ``self`` references resolve to
the composite widget.
"""

import logging
import re

import numpy as np
import pyqtgraph as pg
from pyqtgraph import Qt, ROI
from pyqtgraph.Qt import QtWidgets

from .display_constants import (
    AA_inv, Th, Chi, Deg,
    x_labels_1D, x_units_1D,
)
from .display_logic import plan_overlay, OverlayAction

logger = logging.getLogger(__name__)


class DisplayPlotMixin:
    """Mixin providing 1D plot, waterfall, slice overlay, and mouse helpers.

    Expects the host widget to expose at least:

    - ``self.ui`` (the Ui_Form instance)
    - ``self.scan``, ``self.frame_ids``, ``self.data_1d``, ``self.data_2d``
    - ``self.idxs``, ``self.idxs_1d``
    - ``self.plot``, ``self.plot_win``, ``self.plot_layout``
    - ``self.wf_widget``, ``self.wf_*`` (waterfall state)
    - ``self.curves``, ``self.legend``, ``self.pos_label``
    - ``self.plot_data``, ``self.plot_data_range``, ``self.frame_names``
    - ``self.plotMethod``, ``self.scale``, ``self.cmap``
    - ``self.overlay``, ``self.binned_data``, ``self.binned_widget``
    - ``self._plot_axis_info``, ``self._last_plot_unit``
    - Methods from DisplayDataMixin: ``get_frames_int_1d``, ``get_colors``,
      ``normalize``, ``show_slice_overlay``
    """

    # ── 1D plot data accumulation ─────────────────────────────────

    def _loaded_1d_overlay_labels(self, idxs, *, max_rows=None):
        """Return loaded frame ids and labels that can produce 1D rows."""
        kept = []
        names = []
        slice_suffix = ''
        try:
            if self.ui.slice.isEnabled() and self.ui.slice.isChecked():
                center = self.ui.slice_center.value()
                width = self.ui.slice_width.value()
                slice_suffix = f' [{center:.1f}\u00b1{width:.1f}]'
        except Exception:
            slice_suffix = ''
        for idx in idxs:
            idx = int(idx)
            frame_1d = self.data_1d.get(idx)
            if frame_1d is None:
                continue
            frame_2d = self.data_2d.get(idx)
            try:
                x, y = self.get_int_1d(frame_1d, frame_2d, idx)
            except Exception:
                logger.debug("overlay label rebuild skipped frame %s",
                             idx, exc_info=True)
                continue
            if x is None or y is None:
                continue
            kept.append(idx)
            names.append(f'{self.scan.name}_{idx}{slice_suffix}')
            if max_rows is not None and len(kept) >= max_rows:
                break
        return kept, names

    def update_plot(self):
        """Updates data in plot frame
        """
        if (self.scan.name == 'null_main') or (len(self.frame_ids) == 0):
            data = (np.arange(100), np.arange(100))
            return data

        # Get 1D data for all frames
        ydata, xdata = self.get_frames_int_1d()

        # 2D-derived axes require 2D data; fall back gracefully if unavailable
        if xdata is None or ydata is None:
            _idx = self.ui.plotUnit.currentIndex()
            _info = (self._plot_axis_info[_idx]
                     if hasattr(self, '_plot_axis_info')
                        and 0 <= _idx < len(self._plot_axis_info)
                     else None)
            needs_2d = ((_info and _info['source'] in ('2d', '1d_2d'))
                        or self.ui.slice.isChecked())
            if needs_2d and getattr(self.scan, 'skip_2d', False):
                try:
                    self.window().statusBar().showMessage(
                        "Chi slicing requires 2D integration (1D Only is enabled).", 4000)
                except Exception:
                    logger.debug("Failed to show status bar message about chi slicing", exc_info=True)
            # Fall back: disable slice, retry with plain 1D
            self.ui.slice.setChecked(False)
            ydata, xdata = self.get_frames_int_1d()
            if xdata is None or ydata is None:
                self.clear_plot_view()
                return

        if self.scan.series_average:
            frame_names = [self.scan.name]
        else:
            frame_names = [f'{self.scan.name}_{i}' for i in self.idxs]

        # When slicing is active, include slice parameters in frame names
        # so the same image with different slice ranges can be overlaid.
        if self.ui.slice.isEnabled() and self.ui.slice.isChecked():
            center = self.ui.slice_center.value()
            width = self.ui.slice_width.value()
            suffix = f' [{center:.1f}\u00b1{width:.1f}]'
            frame_names = [n + suffix for n in frame_names]

        # Subtract background
        if self.bkg_1d is not None:
            ydata -= self.bkg_1d
        if ydata.ndim == 1:
            ydata = ydata[np.newaxis, :]

        # Single-mode overplots whatever rows are in the current
        # selection: a single-click gives one curve, shift/cmd-click
        # gives several.  Live-scan selection accumulation (which used
        # to silently turn Single into a Waterfall) is now prevented at
        # the source by ClearAndSelect in h5viewer + latest_frame, so we
        # don't need to narrow ydata here.

        current_plot_unit = self.ui.plotUnit.currentIndex()
        unit_changed = current_plot_unit != self._last_plot_unit
        self._last_plot_unit = current_plot_unit

        # In Overlay/Waterfall: accumulate new frames, skip duplicates.
        # On a unit change, rebuild the accumulated set in the new unit
        # instead of appending across incompatible x grids or dropping all
        # prior curves.  In Single/Sum/Average: always replace with the
        # current selection.  Stage 4: the accumulate/rebuild/replace choice
        # is the pure plan_overlay (unit-tested headlessly); the branch
        # bodies below still own the array work + eviction filtering.
        current_method = self.ui.plotMethod.currentText()
        overlay_action, _ = plan_overlay(
            current_method, unit_changed,
            has_existing=len(self.overlaid_idxs) > 0,
            new_ids=tuple(self.idxs_1d),
            prev_overlaid_ids=tuple(self.overlaid_idxs),
        )

        if overlay_action is OverlayAction.REBUILD:
            rebuild_idxs = list(self.overlaid_idxs)
            y_new, x_new = self.get_frames_int_1d(rebuild_idxs)
            if x_new is not None and y_new is not None:
                if self.bkg_1d is not None:
                    y_new = y_new - self.bkg_1d
                if y_new.ndim == 1:
                    y_new = y_new[np.newaxis, :]
                self.plot_data = [x_new, y_new]
                kept, kept_names = self._loaded_1d_overlay_labels(
                    rebuild_idxs, max_rows=y_new.shape[0],
                )
                self.overlaid_idxs = kept
                self.frame_names = kept_names
            else:
                self.plot_data = [xdata, ydata]
                self.frame_names = list(frame_names)
                self.overlaid_idxs = list(self.idxs_1d)
        elif overlay_action is OverlayAction.APPEND:
            for idx, frame_name, row in zip(self.idxs_1d, frame_names, ydata):
                if frame_name not in self.frame_names:
                    old_x = self.plot_data[0]
                    if old_x.shape == xdata.shape and np.allclose(old_x, xdata):
                        # Same grid — just append
                        self.plot_data[1] = np.vstack((self.plot_data[1], row))
                    else:
                        # Different grid — merge and interpolate.
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
                    self.frame_names.append(frame_name)
                    self.overlaid_idxs.append(int(idx))
        else:
            # Fresh start: Single/Sum/Average, unit changed, or no existing data
            self.plot_data = [xdata, ydata]
            self.frame_names = list(frame_names)
            self.overlaid_idxs = list(self.idxs_1d)

        xdata, ydata = self.plot_data
        if xdata.size == 0 or ydata.size == 0:
            return
        self.plot_data_range = [
            [np.nanmin(xdata), np.nanmax(xdata)],
            [np.nanmin(ydata), np.nanmax(ydata)],
        ]

        self.update_plot_view()

    def _on_plotMethod_changed(self):
        """Handle plotMethod combo box changes.

        Emits ``sigPlotMethodChanged`` so external widgets (e.g. the
        H5Viewer's data list) can adapt their selection mode.

        Switching to Single mode resets accumulated plot data so the
        plot rebuilds from the current selection. Overlay/Waterfall and
        Sum/Average all rely on the user's full multi-selection in
        listData and route through update_plot() so the active set is
        re-aggregated each time.
        """
        new_method = self.ui.plotMethod.currentText()
        try:
            self.sigPlotMethodChanged.emit(new_method)
        except Exception:
            logger.debug("sigPlotMethodChanged emit failed", exc_info=True)

        if getattr(self, 'viewer_mode', None) == 'xye':
            if new_method in ('Single', 'Sum', 'Average'):
                self.clear_overlay()
            self._update_xye_viewer()
            return

        if new_method == 'Single':
            # Reset accumulated data — rebuild from current selection
            self.plot_data = [np.array([]), np.array([])]
            self.frame_names = []
            self.overlaid_idxs = []
            self.update_plot()
        elif new_method in ('Sum', 'Average'):
            # No accumulation needed: aggregation happens inside
            # update_1d_view() based on the current selection.
            self.plot_data = [np.array([]), np.array([])]
            self.frame_names = []
            self.overlaid_idxs = []
            self.update_plot()
        else:
            # Overlay / Waterfall: keep existing accumulated curves and
            # just refresh the rendered view.
            self.update_plot_view()

    def _current_plot_axis_label(self):
        """Return the bottom-axis label and unit for the current 1D view."""
        if getattr(self, 'viewer_mode', None) == 'xye':
            axis = getattr(self, '_viewer_x_axis_label', None)
            if axis is not None:
                return axis

        plot_text = self.ui.plotUnit.currentText()
        m = re.match(r'^(.+?)\s*\((.+)\)$', plot_text)
        if m:
            return m.group(1).strip(), m.group(2).strip()
        return plot_text, ''

    # ── 1D plot view rendering ────────────────────────────────────

    def update_plot_view(self):
        """Updates 1D view of data in plot frame
        """
        if (len(self.frame_ids) == 0) or len(self.data_1d) == 0:
            return

        # Clear curves
        [curve.clear() for curve in self.curves]
        self.curves.clear()

        self.plotMethod = self.ui.plotMethod.currentText()

        self.ui.yOffset.setEnabled(False)
        # yOffset is only meaningful when more than one curve is on screen.
        # That's Overlay with a multi-selection, or Single mode with a
        # manual shift/cmd-click multi-selection (which behaves like Overlay).
        if (self.plotMethod in ('Overlay', 'Single')
                and len(self.frame_names) > 1):
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

        # Overlay/Waterfall always stack the selection.  Single also
        # stacks when the user has multi-selected frames (shift/cmd-click) —
        # that branch behaves like Overlay but without the yOffset.
        multi_single = self.plotMethod == 'Single' and ydata.shape[0] > 1
        if self.plotMethod in ['Overlay', 'Waterfall'] or multi_single:
            ydata = ydata[self.wf_start:self.wf_stop:self.wf_step]
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

        _xl, _xu = self._current_plot_axis_label()
        self.plot.setLabel("bottom", _xl, units=_xu)
        self.plot.setLabel("left", ylabel)

        return s_xdata, s_ydata

    # ── Waterfall rendering ───────────────────────────────────────

    def _wf_y_axis(self, n_rows: int):
        """Compute the waterfall y-axis array.

        G1: shared between :meth:`update_wf` and
        :meth:`update_wf_pmesh`.  Returns the ydata array shaped to
        ``n_rows`` (= data.shape[0] after the wf_start/wf_step slice)
        or ``None`` when a metadata key is missing — caller handles
        the no-axis case.

        ``self.wf_yaxis`` selects the source:

        * ``'Frame #'`` → ``arange`` rooted at ``wf_start + 1``.
        * ``'Time (s)'`` / ``'Time (minutes)'`` →
          ``scan_info['epoch']`` minus its own minimum (relative).
        * anything else → ``scan_info[wf_yaxis]`` directly.

        For everything but ``'Frame #'`` we lift the values from
        ``self.data_1d[idx].scan_info`` for every idx in
        ``self.idxs``, then slice with the same wf_start/wf_step the
        data uses.
        """
        if self.wf_yaxis == 'Frame #':
            return np.asarray(np.arange(n_rows) + self.wf_start + 1,
                              dtype=float)
        try:
            if self.wf_yaxis == 'Time (s)':
                s_ydata = np.asarray(
                    [self.data_1d[idx].scan_info['epoch']
                     for idx in self.idxs]
                )
                s_ydata -= s_ydata.min()
            elif self.wf_yaxis == 'Time (minutes)':
                s_ydata = np.asarray(
                    [self.data_1d[idx].scan_info['epoch']
                     for idx in self.idxs]
                ) / 60.
                s_ydata -= s_ydata.min()
            else:
                s_ydata = np.asarray(
                    [self.data_1d[idx].scan_info[self.wf_yaxis]
                     for idx in self.idxs]
                )
            return s_ydata[self.wf_start:self.wf_stop:self.wf_step]
        except KeyError as e:
            logger.debug('Counter %s not present in metadata: %s',
                         self.wf_yaxis, e)
            return None

    def update_wf(self):
        """Updates data in 1D plot Frame
        """
        self.setup_wf_layout()

        xdata_, data_ = self.plot_data
        s_xdata, data = xdata_.copy(), data_.copy()
        data = data[self.wf_start:self.wf_stop:self.wf_step, :]

        s_ydata = self._wf_y_axis(data.shape[0])
        if s_ydata is None:
            return

        from ...gui_utils import get_rect
        rect = get_rect(s_xdata, s_ydata)

        self.wf_widget.setImage(data.T, scale=self.scale, cmap=self.cmap)
        self.wf_widget.setRect(rect)

        _xl, _xu = self._current_plot_axis_label()
        self.wf_widget.image_plot.setLabel("bottom", _xl, units=_xu)
        self.wf_widget.image_plot.setLabel("left", self.wf_yaxis)

    def update_wf_pmesh(self):
        """Updates data in 1D plot Frame (pcolormesh waterfall rendering)
        """
        self.setup_wf_layout()

        xdata_, data_ = self.plot_data
        s_xdata, data = xdata_.copy(), data_.copy()
        data = data[self.wf_start:self.wf_stop:self.wf_step, :]

        x_max, x_min = np.max(s_xdata), np.min(s_xdata)
        x_step = (x_max - x_min)/len(s_xdata)
        s_xdata = np.append(s_xdata, [x_max + x_step])
        s_xdata -= x_step/2
        s_xdata = np.tile(s_xdata, (data.shape[0]+1, 1)).T

        s_ydata = self._wf_y_axis(data.shape[0])
        if s_ydata is None:
            return

        y_max, y_min = np.max(s_ydata), np.min(s_ydata)
        y_step = (y_max - y_min)/len(s_ydata)
        s_ydata = np.append(s_ydata, [y_max + y_step])
        s_ydata -= y_step/2.
        s_ydata = np.tile(s_ydata, (data.shape[1]+1, 1))

        levels = np.nanpercentile(data, (1, 98))
        self.wf_widget.imageItem.setLevels(levels)
        self.wf_widget.imageItem.setData(s_xdata, s_ydata, data.T)
        self.wf_widget.imageItem.informViewBoundsChanged()

        from ...gui_utils import get_rect
        rect = get_rect(s_xdata[:, 0], s_ydata[0])
        self.wf_widget.setRect(rect)

        if getattr(self, 'viewer_mode', None) == 'xye':
            _xl, _xu = self._current_plot_axis_label()
        else:
            plotUnit = self.ui.plotUnit.currentIndex()
            _xl, _xu = x_labels_1D[plotUnit], x_units_1D[plotUnit]
        self.wf_widget.image_plot.setLabel("bottom", _xl, units=_xu)
        self.wf_widget.image_plot.setLabel("left", self.wf_yaxis)

        return data

    # ── Curve / layout helpers ────────────────────────────────────

    def setup_curves(self):
        """Initialize curves for line plots
        """
        self.curves.clear()
        self.legend.clear()

        frame_ids = self.frame_names[self.wf_start:self.wf_stop:self.wf_step]
        if (self.plotMethod in ['Sum', 'Average'] and
                len(self.frame_names) > 1):
            frame_ids = f'{self.plotMethod} [{self.frame_names[0]}'
            for frame_name in self.frame_names[1:]:
                frame_ids += f', {frame_name}'
            frame_ids = [frame_ids + ']']

        colors = self.get_colors()
        self.curves = [self.plot.plot(
            pen=color,
            symbolBrush=color,
            symbolPen=color,
            symbolSize=4,
            name=frame_id,
        ) for (color, frame_id) in zip(colors, frame_ids)]

        if not self.ui.showLegend.isChecked():
            self.legend.clear()

    def clear_1D(self):
        """Initialize curves for line plots
        """
        self.frame_names.clear()
        self.frame_ids.clear()
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

    def setup_1d_layout(self):
        """Setup the layout for 1D plot
        """
        self.wf_widget.setParent(None)
        self.plot_layout.addWidget(self.plot_win)

        # Options is always reachable now — it holds Legend + Overlay Offset
        # (the waterfall y-axis/start/step inside grey out when not in a
        # waterfall).  Was disabled in single-curve mode, which would have
        # hidden the Legend toggle.
        self.ui.wf_options.setEnabled(True)
        if len(self.plot_data[1]) > 1:
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

    # ── Waterfall options popup ───────────────────────────────────

    def popup_wf_options(self):
        """
        Popup Qt Window to select options for Waterfall Plot
        Options include Y-axis unit and number of points to skip
        """
        if self.wf_dialog.layout() is None:
            self.setup_wf_options_widget()

        self.wf_dialog.show()

    def setup_wf_options_widget(self):
        """Build the 1D-plot Options dialog, grouped into three sections:

        * **Waterfall** — y-axis source (Frame #/Time/metadata), Start, Stop,
          Step.  Govern which frames map onto the waterfall image and its
          y-axis.
        * **Overlay** — Offset (vertical stacking between curves).
        * **Legend** — show/hide the curve legend.
        """
        layout = QtWidgets.QVBoxLayout()
        self.wf_dialog.setLayout(layout)

        def _section(title):
            box = QtWidgets.QGroupBox(title)
            grid = QtWidgets.QGridLayout()
            box.setLayout(grid)
            layout.addWidget(box)
            return grid

        # ── Waterfall ─────────────────────────────────────────────
        wf = _section('Waterfall')
        wf.addWidget(QtWidgets.QLabel('Y-Axis'), 0, 0)
        wf.addWidget(QtWidgets.QLabel('Start'), 0, 1)
        wf.addWidget(QtWidgets.QLabel('Stop'), 0, 2)
        wf.addWidget(QtWidgets.QLabel('Step'), 0, 3)
        wf.addWidget(self.wf_yaxis_widget, 1, 0)
        wf.addWidget(self.wf_start_widget, 1, 1)
        wf.addWidget(self.wf_stop_widget, 1, 2)
        wf.addWidget(self.wf_step_widget, 1, 3)

        # ── Overlay ───────────────────────────────────────────────
        ov = _section('Overlay')
        ov.addWidget(QtWidgets.QLabel('Offset'), 0, 0)
        ov.addWidget(self.ui.yOffset, 0, 1)

        # ── Legend ────────────────────────────────────────────────
        lg = _section('Legend')
        lg.addWidget(self.ui.showLegend, 0, 0)

        # ── Dialog buttons ────────────────────────────────────────
        btns = QtWidgets.QHBoxLayout()
        btns.addStretch(1)
        btns.addWidget(self.wf_accept_button)
        btns.addWidget(self.wf_cancel_button)
        layout.addLayout(btns)

        frame = self.data_1d[self.idxs_1d[0]]
        counters = list(frame.scan_info.keys())
        counters = ['Frame #', 'Time (s)', 'Time (minutes)'] + counters
        self.wf_yaxis_widget.addItems(counters)

        self.wf_start_widget.setDecimals(0)
        self.wf_start_widget.setRange(1, 100000)
        self.wf_start_widget.setValue(1)

        # Stop: 0 is the sentinel for "through the last frame".
        self.wf_stop_widget.setDecimals(0)
        self.wf_stop_widget.setRange(0, 100000)
        self.wf_stop_widget.setValue(0)
        self.wf_stop_widget.setSpecialValueText('End')  # shown when value==0

        self.wf_step_widget.setDecimals(0)
        self.wf_step_widget.setRange(1, 1000)

        self.wf_accept_button.clicked.connect(self.get_wf_option)
        self.wf_cancel_button.clicked.connect(self.close_wf_popup)

    def get_wf_option(self):
        self.wf_yaxis = self.wf_yaxis_widget.currentText()

        self.wf_start = int(self.wf_start_widget.value()) - 1
        # Stop: 0 (shown as "End") → None = slice through the last frame;
        # otherwise it's a 1-based inclusive end (→ exclusive Python stop).
        _stop = int(self.wf_stop_widget.value())
        self.wf_stop = None if _stop <= 0 else _stop
        self.wf_step = int(self.wf_step_widget.value())

        self.close_wf_popup()
        self.update_plot_view()

    def close_wf_popup(self):
        self.wf_dialog.close()

    # ── Mouse tracking ────────────────────────────────────────────

    def trackMouse(self):
        """Set up mouse tracking on the plot scene.

        N6: pre-N6 this both passed ``slot=self.mouseMoved`` to the
        SignalProxy constructor AND called
        ``proxy.signal.connect(self.mouseMoved)`` afterwards —
        ``mouseMoved`` ran twice per mouse event.  Also: the proxy
        was a local variable that risked garbage-collection (Qt
        signals don't keep proxies alive on their own).  Stash it
        on ``self`` and use the constructor connection only.
        """
        self._mouse_proxy = pg.SignalProxy(
            signal=self.plot.scene().sigMouseMoved,
            rateLimit=60,
            slot=self.mouseMoved,
        )

    def mouseMoved(self, pos):
        """Slot wired to ``pg.SignalProxy(sigMouseMoved)``.

        SignalProxy wraps emissions in a tuple — the slot receives
        ``(QPointF,)`` rather than the bare event.  Unpack defensively
        so the same slot also accepts a direct ``QPointF`` (in case
        a future caller wires up sigMouseMoved without the proxy).
        """
        from PySide6.QtCore import Qt as pyQt
        # SignalProxy delivers args as a single-element tuple; the
        # bare-signal path delivers the QPointF directly.  Handle both.
        if isinstance(pos, tuple):
            if not pos:
                return
            pos = pos[0]

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

    # ── Slice overlay ─────────────────────────────────────────────

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
        if self.scan.gi or (info and info['source'] not in ('2d', '1d_2d')):
            self.update_plot()
            return

        # Standard mode, chi axis: handle Q ↔ 2θ conversion
        if not self.scan.gi and info and info.get('axis') == 'azimuthal':
            imageUnit = self.ui.imageUnit.currentIndex()
            cen = self.ui.slice_center.value()
            wid = self.ui.slice_width.value()
            _range = np.array([cen - wid, cen + wid])

            try:
                frame_for_wl = self.data_1d[self.idxs_1d[0]]
            except (IndexError, KeyError):
                self.update_plot()
                return
            wavelength = self._get_wavelength(frame_for_wl)
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
