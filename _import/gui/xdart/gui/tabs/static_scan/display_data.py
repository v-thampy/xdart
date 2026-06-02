# -*- coding: utf-8 -*-
"""Data fetching, processing, and export methods for displayFrameWidget.

This mixin extracts ~500 lines of data-access logic from the monolithic
displayFrameWidget class.  Methods here deal with reading data from
LiveScan / LiveFrame containers, normalization, unit conversion,
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

    - ``self.scan``, ``self.frame``, ``self.frames``
    - ``self.frame_ids``, ``self.data_1d``, ``self.data_2d``
    - ``self.data_lock`` (threading.RLock guarding the dicts)
    - ``self.idxs``, ``self.idxs_1d``, ``self.idxs_2d``, ``self.overall``
    - ``self.ui`` (the Ui_Form instance)
    - ``self.normChannel``, ``self.bkg_*``
    - ``self._plot_axis_info``

    Locking contract (M3)
    =====================

    ``data_1d`` and ``data_2d`` are shared between the wrangler
    thread, the integrator thread, the fileHandlerThread / the M1
    LoadFramesWorker, and the GUI thread.  All **mutating** access
    (assignment, ``del``, ``clear``, ``pop``) goes through
    ``self.data_lock``.  **Read** access either takes ``data_lock``
    explicitly (when iterating multiple keys) or uses
    :meth:`_snapshot_data` to grab a stable view to iterate
    lock-free.

    The CPython GIL makes single ``dict[k]`` lookups atomic, so
    direct cache-hit reads like ``self.data_1d.get(idx)`` are still
    safe.  The unsafe pattern was *iterating* the dicts (``for k, v
    in self.data_2d.items():`` or ``list(self.data_2d.keys())``)
    without a lock — a concurrent writer can mutate the dict
    mid-iteration and raise RuntimeError ("dictionary changed size
    during iteration").  Use ``_snapshot_data`` for those paths.
    """

    @staticmethod
    def _sanitize_display_image(data):
        """Return a float image with detector sentinels masked to NaN."""
        arr = np.asarray(data, dtype=float)
        bad = ~np.isfinite(arr) | (arr >= 4294967295.0)
        if arr.size and np.isfinite(arr).any():
            # Some 16-bit detector/raw readers preserve invalid pixels as the
            # uint16 ceiling.  If enough pixels sit exactly at 65535, treat
            # that ceiling as a display sentinel so autoscale uses the real
            # image range instead of rendering the frame nearly black.
            finite = np.isfinite(arr)
            sentinel16 = finite & (arr == 65535.0)
            if sentinel16.any() and sentinel16.sum() / arr.size > 1e-4:
                bad |= sentinel16
        if bad.any():
            arr = arr.copy()
            arr[bad] = np.nan
        return arr

    def _snapshot_data(self, idxs):
        """Return a small {idx: (frame_1d, frame_2d_dict)} dict for
        the requested ``idxs``, sampled atomically under
        ``self.data_lock``.

        M3 helper: callers that need to iterate over a set of
        frames' data (e.g. for averaging) should take the snapshot
        once and then process it without holding the lock.  This
        keeps the lock window short (one dict-comprehension's worth)
        and avoids the "dictionary changed size during iteration"
        race that the wrangler thread can otherwise trigger.

        Missing frames are silently omitted from the result — the
        caller is expected to handle partial data anyway.
        """
        with self.data_lock:
            return {
                int(idx): (
                    self.data_1d.get(int(idx)),
                    self.data_2d.get(int(idx), {}),
                )
                for idx in idxs
            }

    # ── Raw 2D data access ────────────────────────────────────────

    def get_frames_map_raw(self, idxs=None, *, prefer_thumbnail=False,
                           return_source=False, require_all=False):
        """Return 2D frame data for multiple frames (averaged).

        Falls back to the stored thumbnail when full-resolution raw data
        is not available (e.g. when loading from NeXus files that only
        store integration results + thumbnails).

        M3: takes a single snapshot of the requested idxs under
        ``data_lock`` and then iterates the snapshot lock-free, so
        a concurrent writer can keep streaming new frames without
        racing the iteration.
        """
        if idxs is None:
            idxs = self.idxs_2d
        idxs = list(idxs)

        snapshot = self._snapshot_data(idxs)

        intensity, ctr = 0., 0
        sources = set()
        sanitize = getattr(
            self, '_sanitize_display_image',
            DisplayDataMixin._sanitize_display_image,
        )
        for nn, idx in enumerate(idxs):
            frame_1d, frame_2d = snapshot.get(int(idx), (None, {}))
            raw = frame_2d.get('map_raw')
            bg = frame_2d.get('bg_raw', 0)
            # Try thumbnail from data_2d, then fall back to data_1d
            thumb = frame_2d.get('thumbnail')
            if thumb is None and frame_1d is not None:
                thumb = getattr(frame_1d, 'thumbnail', None)
            if prefer_thumbnail and thumb is not None:
                raw = thumb
                bg = 0
                source = 'thumbnail'
            elif raw is not None:
                source = 'raw'
            elif thumb is not None:
                source = 'thumbnail'
            else:
                source = None
            # F1: was `for kk in range(3): try: ...; break; except
            # ValueError: time.sleep(0.5)`.  The retry/sleep pattern
            # was running on the Qt thread — visible UI freeze on
            # any single broken frame.  The ValueError originated from
            # shape mismatches during early-load races; we now log
            # them at debug and move on (the GUI re-fires update
            # signals on its own when the wrangler finishes more
            # frames, so a missed average will be recomputed next
            # cycle).
            try:
                scan_info = frame_1d.scan_info if frame_1d is not None else {}
                if raw is not None:
                    raw_data = sanitize(raw)
                    intensity += self.normalize(raw_data - bg, scan_info)
                    ctr += 1
                    sources.add(source or 'raw')
                elif thumb is not None:
                    # Use thumbnail as fallback when raw isn't stored
                    thumb_data = sanitize(thumb)
                    intensity += self.normalize(
                        thumb_data, scan_info)
                    ctr += 1
                    sources.add('thumbnail')
            except ValueError as e:
                logger.debug(
                    "get_frames_map_raw skipped frame %s due to shape "
                    "mismatch: %s", idx, e,
                )

        if require_all and ctr != len(idxs):
            if return_source:
                return None, None
            return None

        if ctr > 0:
            intensity /= ctr
        else:
            if return_source:
                return None, None
            return None

        if len(sources) == 1:
            source = next(iter(sources))
        else:
            source = 'mixed'
        data = np.asarray(intensity, dtype=float)
        if return_source:
            return data, source
        return data

    # G2: get_scan_map_raw was deleted.  It read scan.overall_raw,
    # an in-memory accumulator that doesn't survive v2 reload (the
    # loader doesn't repopulate it) and goes stale under R1's
    # replace-frames save.  The Overall view in update_image now
    # aggregates via get_frames_map_raw(list(scan.frames.index)).

    # ── 2D integration data access ────────────────────────────────

    def get_frames_int_2d(self, idxs=None, *, require_all=False):
        """Return 2D frame data for multiple frames (averaged).

        Mirrors :meth:`get_frames_map_raw` / :meth:`get_frames_int_1d`:
        accumulates per-frame normalized intensity from ``data_2d`` and
        averages on the fly. No external state required — always reflects
        the current selection in ``data_2d``.

        Returns ``(intensity, xdata, ydata)`` or ``(None, None, None)``
        if nothing usable is loaded.

        M3: uses ``_snapshot_data`` for a stable view of the
        requested idxs; concurrent writes to ``data_1d``/``data_2d``
        no longer race this iteration.
        """
        if idxs is None:
            idxs = self.idxs_2d
        idxs = list(idxs)

        if not idxs:
            return None, None, None

        snapshot = self._snapshot_data(idxs)

        intensity = None
        xdata = ydata = None
        ctr = 0
        for idx in idxs:
            frame_1d, frame_2d = snapshot.get(int(idx), (None, None))
            if frame_2d is None or frame_2d.get('int_2d') is None:
                continue
            _gi2d = frame_2d.get('gi_2d', {})
            try:
                _i = self.get_int_2d(frame_2d['int_2d'], frame_1d, gi_2d=_gi2d)
            except (ValueError, AttributeError, TypeError):
                continue
            if _i.ndim != 2:
                continue
            if intensity is None:
                intensity = np.asarray(_i, dtype=float)
                xdata, ydata = self.get_xydata(
                    frame_2d['int_2d'], gi_2d=_gi2d, frame=frame_1d)
            else:
                try:
                    intensity = intensity + _i
                except (ValueError, AttributeError, TypeError):
                    continue
            ctr += 1

        if require_all and ctr != len(idxs):
            return None, None, None

        if intensity is None or ctr == 0:
            return None, None, None

        intensity = intensity / ctr
        return intensity, xdata, ydata

    # G2: get_scan_int_2d was deleted.  It read scan.bai_2d, an
    # in-memory accumulator that doesn't survive v2 reload.  The
    # Overall view in update_binned now uses
    # get_frames_int_2d(list(scan.frames.index)).  The comment
    # at the call site (display_frame_widget.update_binned) already
    # noted this path returned 1×1 zeros for NeXus files — so it's
    # been functionally dead since v2 landed.

    def get_int_2d(self, int_2d, frame_1d=None, normalize=True, gi_2d=None):
        """Returns the appropriate 2D data depending on the chosen axes.
        In GI mode, int_2d already holds the selected mode's data.
        """
        if int_2d is None:
            return np.zeros((1, 1))
        # int_2d is always the correct result (GI or standard)
        intensity_2d = int_2d.intensity
        intensity = np.asarray(intensity_2d.copy(), dtype=float)

        if normalize:
            if frame_1d is not None:
                intensity = self.normalize(intensity, frame_1d.scan_info)
            else:
                norm_fac = len(self.scan.frames.index)
                if self.normChannel:
                    norm = self.scan.scan_data[self.normChannel].sum()
                    if norm > 0:
                        norm_fac = norm
                intensity /= norm_fac

        return intensity

    # ── 1D integration data access ────────────────────────────────

    def get_frames_int_1d(self, idxs=None, rv='all'):
        """Return 1D data for multiple frames"""
        if idxs is None:
            idxs = self.idxs_1d

        # Collect rows then stack once.  The previous code re-allocated a
        # growing array with np.vstack on every iteration — O(N^2) over the
        # frames of a Waterfall/Overlay/Sum/Average; one stack at the end is
        # O(N).
        xdata = None
        ys: list = []
        for idx in idxs:
            frame_1d = self.data_1d.get(int(idx), None)
            if frame_1d is None:
                continue
            frame_2d = self.data_2d.get(int(idx), None)
            x, y = self.get_int_1d(frame_1d, frame_2d, idx)
            if x is None or y is None:
                continue
            if xdata is None:
                xdata = x
            ys.append(y)

        if not ys:
            return None, None

        ydata = ys[0] if len(ys) == 1 else np.vstack(ys)

        if ydata.ndim == 2:
            if rv == 'average':
                ydata = np.nanmean(ydata, 0)
            elif rv == 'sum':
                ydata = np.nansum(ydata, 0)

        return ydata, xdata

    def get_int_1d(self, frame, frame_2d, idx):
        """Returns 1D integrated data for frame.

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
            int_1d = frame.int_1d
            if int_1d is None:
                return None, None
            intensity = int_1d.intensity
            ydata = self.normalize(intensity, frame.scan_info)
            xdata = self.get_xdata(frame)
            return xdata, ydata

        # --- 2D path: project from 2D map ---
        if frame_2d is None:
            return None, None

        intensity = self.get_int_2d(frame_2d['int_2d'], frame, normalize=False,
                                    gi_2d=frame_2d.get('gi_2d', {}))
        if intensity.ndim < 2:
            return None, None

        _i2d = frame_2d['int_2d']
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

        ydata = self.normalize(ydata, frame.scan_info)
        return xdata, ydata

    # ── Axis data helpers ─────────────────────────────────────────

    def get_xydata(self, int_2d, gi_2d=None, frame=None):
        """Reads the 2D unit box and returns the appropriate radial / azimuthal
        axes, converting Q ↔ 2θ on the fly when the selected ``imageUnit``
        differs from the integration unit (mirrors :meth:`get_xdata` for the
        1D plot — without this the cake's x-axis stayed in Q while only the
        label switched to 2θ).

        In GI mode ``int_2d`` already holds the selected GI-mode result
        (qz/qxy or q/χ), so the axes are returned as-is — there's no
        Q↔2θ toggle there (the imageUnit combo is fixed/disabled in GI).

        args:
            int_2d: IntegrationResult2D, primary integration result
            gi_2d: dict of IntegrationResult2D for GI modes (unused, kept
                   for API compatibility)
            frame: optional LiveFrame for the wavelength lookup

        returns:
            xdata, ydata: numpy arrays for radial and azimuthal axes.
        """
        if int_2d is None:
            return np.array([]), np.array([])
        radial = np.asarray(int_2d.radial, dtype=float)
        azimuthal = int_2d.azimuthal
        if getattr(self.scan, 'gi', False):
            return radial, azimuthal

        from .display_constants import AA_inv, Th

        image_label = self.ui.imageUnit.currentText()
        data_unit = getattr(int_2d, 'unit', 'q_A^-1')
        want_tth = (Th in image_label)         # imageUnit label contains θ
        have_tth = ('2th' in data_unit)
        if want_tth and not have_tth:
            wl = self._get_wavelength(frame)
            if wl and wl > 0:
                lam_A = wl * 1e10
                arg = np.clip(radial * lam_A / (4 * np.pi), -1, 1)
                radial = 2 * np.degrees(np.arcsin(arg))
        elif not want_tth and have_tth and (AA_inv in image_label):
            wl = self._get_wavelength(frame)
            if wl and wl > 0:
                lam_A = wl * 1e10
                radial = (4 * np.pi / lam_A) * np.sin(np.radians(radial / 2))
        return radial, azimuthal

    def get_xdata(self, frame):
        """Reads the unit box and returns appropriate xdata for 1D plot.

        Handles on-the-fly Q ↔ 2θ conversion when the plotUnit selection
        differs from the integration unit stored in int_1d.

        args:
            frame: LiveFrame copy (data_1d entry) holding int_1d and gi_1d

        returns:
            xdata: numpy array, x axis data for plot.
        """
        from .display_constants import AA_inv, Th

        int_1d = getattr(frame, 'int_1d', None)
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
            wl = self._get_wavelength(frame)
            if wl and wl > 0:
                lam_A = wl * 1e10
                arg = np.clip(radial * lam_A / (4 * np.pi), -1, 1)
                return 2 * np.degrees(np.arcsin(arg))
        elif not want_tth and have_tth and (AA_inv in plot_label):
            # Data is in 2θ, display wants Q: convert 2θ → Q
            wl = self._get_wavelength(frame)
            if wl and wl > 0:
                lam_A = wl * 1e10
                return (4 * np.pi / lam_A) * np.sin(np.radians(radial / 2))

        return radial

    def _get_wavelength(self, frame=None):
        """Return the X-ray wavelength in metres.

        Tries several sources in order:
        1. ``frame.integrator.wavelength`` (available during live processing)
        2. ``self.scan.mg_args['wavelength']`` (persisted in NXS)
        3. The calibration group in the HDF5 file

        Returns None if the wavelength cannot be determined.
        """
        # 1. From the frame's integrator (fastest, works during live runs)
        if frame is not None:
            ai = getattr(frame, 'integrator', None)
            wl = getattr(ai, 'wavelength', None) if ai else None
            if wl and wl > 0:
                return wl

        # 2. From scan.mg_args (loaded when NXS is opened)
        wl = self.scan.mg_args.get('wavelength', None) if hasattr(self.scan, 'mg_args') else None
        if wl and wl > 0:
            return wl

        # 3. Read from the HDF5 calibration group
        try:
            import h5py
            with h5py.File(self.scan.data_file, 'r') as f:
                wl = float(f['entry/calibration/wavelength'][()]) # type: ignore
                if wl > 0:
                    return wl
        except Exception:
            logger.debug("Failed to read wavelength from HDF5 calibration group in %s", self.scan.data_file, exc_info=True)

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
            scan_data_keys = self.scan.scan_data.columns
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
            more_colors = colors_tuples(np.linspace(0, 1, len(self.frame_names)))
            colors = np.vstack((colors, more_colors[:, 0:3]))

        else:
            try:
                colors_tuples = plt.get_cmap(self.cmap)
            except ValueError:
                colors_tuples = plt.get_cmap('jet', 256)
            colors = colors_tuples(np.linspace(0, 1, len(self.frame_names)))[:, 0:3]

        colors = np.round(colors * [255, 255, 255]).astype(int)
        colors = [tuple(color[:3]) for color in colors]

        return colors

    # ── Stubs for future implementation ───────────────────────────

    def get_profile_chi(self, frame):
        """Extract intensity profile along chi from frame.

        Args:
            frame: LiveFrame object with 2D integration data.

        Returns:
            ndarray: Intensity integrated along chi over the Q range
                     specified by the UI slice controls.

        .. note:: Not yet implemented.
        """
        raise NotImplementedError("get_profile_chi is not yet implemented")

    def get_chi_1d(self, frame):
        """Extract 1D chi profile from frame.

        Args:
            frame: LiveFrame object with 2D integration data.

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

        fname = f'{self.scan.name}'
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
            path = os.path.dirname(self.scan.data_file)
            path = os.path.join(path, self.scan.name)
            Path(path).mkdir(parents=True, exist_ok=True)

        fname = os.path.join(path, fname)

        xdata, ydata = self.plot_data
        # H4: Average / Sum produces ONE combined output.  Pre-H4 the
        # code wrote the combined file AND then fell through to the
        # per-frame loop below — silently producing dozens of extra
        # per-frame .xye files alongside an "average.xye" in the
        # same directory.  Branch cleanly: Average/Sum → combined
        # only; everything else (Overlay, Single, Waterfall) →
        # per-frame files.
        if self.plotMethod in ('Average', 'Sum'):
            if self.plotMethod == 'Average':
                s_ydata = np.nanmean(ydata, 0)
            else:
                s_ydata = np.nansum(ydata, 0)
            xye_fname = f'{fname}.xye'
            ut.write_xye(xye_fname, xdata, s_ydata)
        else:
            idxs = [
                frame.replace(f'{self.scan.name}_', '')
                for frame in self.frame_names
            ]
            for s_ydata, idx in zip(ydata, idxs):
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
