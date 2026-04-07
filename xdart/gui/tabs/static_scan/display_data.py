# -*- coding: utf-8 -*-
"""Data fetching, processing, and export methods for displayFrameWidget.

This mixin extracts ~500 lines of data-access logic from the monolithic
displayFrameWidget class.  Methods here deal with reading data from
EwaldSphere / EwaldArch containers, normalization, unit conversion,
colour generation, and saving results to disk.

The mixin is designed to be inherited by displayFrameWidget alongside
QWidget, so all ``self`` references resolve to the composite widget.
"""

import logging
import os
import time

import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class DisplayDataMixin:
    """Mixin providing data-fetching, processing, and export helpers.

    Expects the host widget to expose at least:

    - ``self.sphere``, ``self.arch``, ``self.arches``
    - ``self.arch_ids``, ``self.data_1d``, ``self.data_2d``
    - ``self.idxs``, ``self.idxs_1d``, ``self.idxs_2d``, ``self.overall``
    - ``self.ui`` (the Ui_Form instance)
    - ``self.normChannel``, ``self.bkg_*``
    - ``self._plot_axis_info``
    """

    # ── Raw 2D data access ────────────────────────────────────────

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

    # ── 2D integration data access ────────────────────────────────

    def get_arches_int_2d(self, idxs=None):
        """Return 2D arch data for multiple arches (averaged).

        Mirrors :meth:`get_arches_map_raw` / :meth:`get_arches_int_1d`:
        accumulates per-arch normalized intensity from ``data_2d`` and
        averages on the fly. No external state required — always reflects
        the current selection in ``data_2d``.

        Returns ``(intensity, xdata, ydata)`` or ``(None, None, None)``
        if nothing usable is loaded.
        """
        if idxs is None:
            idxs = self.idxs_2d

        if not idxs:
            return None, None, None

        intensity = None
        xdata = ydata = None
        ctr = 0
        for idx in idxs:
            arch_1d = self.data_1d.get(int(idx))
            arch_2d = self.data_2d.get(int(idx))
            if arch_2d is None or arch_2d.get('int_2d') is None:
                continue
            _gi2d = arch_2d.get('gi_2d', {})
            try:
                _i = self.get_int_2d(arch_2d['int_2d'], arch_1d, gi_2d=_gi2d)
            except (ValueError, AttributeError, TypeError):
                continue
            if _i.ndim != 2:
                continue
            if intensity is None:
                intensity = np.asarray(_i, dtype=float)
                xdata, ydata = self.get_xydata(arch_2d['int_2d'], gi_2d=_gi2d)
            else:
                try:
                    intensity = intensity + _i
                except (ValueError, AttributeError, TypeError):
                    continue
            ctr += 1

        if intensity is None or ctr == 0:
            return None, None, None

        intensity = intensity / ctr
        return intensity, xdata, ydata

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

    # ── 1D integration data access ────────────────────────────────

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

    # ── Axis data helpers ─────────────────────────────────────────

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

    def get_xdata(self, arch):
        """Reads the unit box and returns appropriate xdata for 1D plot.

        Handles on-the-fly Q ↔ 2θ conversion when the plotUnit selection
        differs from the integration unit stored in int_1d.

        args:
            arch: EwaldArch copy (data_1d entry) holding int_1d and gi_1d

        returns:
            xdata: numpy array, x axis data for plot.
        """
        from .display_constants import AA_inv, Th

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

    # ── Normalization ─────────────────────────────────────────────

    def normalize(self, int_data, scan_info):
        """Normalize intensity data by the selected normalization channel.

        args:
            int_data: numpy array, intensity data to normalize
            scan_info: dict, metadata containing normalization counters

        returns:
            intensity: numpy array, normalized data
        """
        try:
            intensity = np.asarray(int_data.copy(), dtype=float)
        except AttributeError:
            return np.zeros((10, 10))

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

    # ── Colour generation ─────────────────────────────────────────

    def get_colors(self):
        """Generate a list of RGB colour tuples for plot curves."""
        import matplotlib.pyplot as plt

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

    # ── Stubs for future implementation ───────────────────────────

    def get_profile_chi(self, arch):
        """Extract intensity profile along chi from arch.

        Args:
            arch: EwaldArch object with 2D integration data.

        Returns:
            ndarray: Intensity integrated along chi over the Q range
                     specified by the UI slice controls.

        .. note:: Not yet implemented.
        """
        raise NotImplementedError("get_profile_chi is not yet implemented")

    def get_chi_1d(self, arch):
        """Extract 1D chi profile from arch.

        Args:
            arch: EwaldArch object with 2D integration data.

        Returns:
            ndarray: 1D intensity vs chi extracted from 2D data.

        .. note:: Not yet implemented.
        """
        raise NotImplementedError("get_chi_1d is not yet implemented")

    # ── Save / Export ─────────────────────────────────────────────

    def save_image(self):
        """Saves currently displayed image. Formats are automatically
        grabbed from Qt. Also implements tiff saving.
        """
        import pyqtgraph
        import pyqtgraph.exporters
        from pyqtgraph.Qt import QtWidgets
        from xdart.utils import split_file_name

        QFileDialog = QtWidgets.QFileDialog

        formats = [
            str(f.data(), encoding='utf-8').lower() for f in
            pyqtgraph.Qt.QtGui.QImageReader.supportedImageFormats()
        ]

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
        import pyqtgraph
        import pyqtgraph.exporters
        from pyqtgraph.Qt import QtWidgets
        import xdart.utils as ut

        QFileDialog = QtWidgets.QFileDialog

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
