# -*- coding: utf-8 -*-
"""
nexusThread — worker thread for NeXus/Tiled wrangler.

Reads frames from a NeXus HDF5 file (Bluesky suitcase-nexus format)
and integrates each one using the same EwaldArch pipeline as specThread.

@author: thampy
"""

# Standard library imports
import os
import time
import numpy as np
from pathlib import Path

# HDF5
import h5py

# Qt imports
from pyqtgraph import Qt

# Project imports
from xdart.modules.ewald import EwaldArch, EwaldSphere
from ssrl_xrd_tools.integrate.gid import create_fiber_integrator
from ssrl_xrd_tools.integrate.calibration import poni_to_integrator, get_detector
from ssrl_xrd_tools.io.nexus import find_nexus_image_dataset, read_nexus
from ssrl_xrd_tools.io.image import read_image
from ssrl_xrd_tools.io.export import write_xye
from xdart.utils import catch_h5py_file as _catch_h5
from xdart.utils.h5pool import get_pool as _get_h5pool
from .wrangler_widget import wranglerThread


class nexusThread(wranglerThread):
    """Thread for processing NeXus/HDF5 image stacks.

    Reads an image dataset from a NeXus file, integrates each frame,
    and emits sigUpdate after each one.

    signals:
        showLabel: str, status text for the UI label
    """
    showLabel = Qt.QtCore.Signal(str)

    def __init__(
            self,
            command_queue,
            sphere_args,
            file_lock,
            fname,
            nexus_file,
            poni,
            mask_file,
            gi,
            th_mtr,
            sample_orientation,
            tilt_angle,
            gi_mode_1d,
            gi_mode_2d,
            command,
            sphere,
            data_1d,
            data_2d,
            entry='entry',
            parent=None):

        super().__init__(command_queue, sphere_args, fname, file_lock, parent)

        self.nexus_file = nexus_file
        self.poni = poni
        self.mask_file = mask_file
        self.gi = gi
        self.th_mtr = th_mtr
        self.sample_orientation = sample_orientation
        self.tilt_angle = tilt_angle
        self.gi_mode_1d = gi_mode_1d
        self.gi_mode_2d = gi_mode_2d
        self.command = command
        self.sphere = sphere
        self.data_1d = data_1d
        self.data_2d = data_2d
        self.entry = entry

        self.detector = None
        self.mask = None
        self._cached_gi_incident_angle = None

        # NeXus processing is always batch-mode equivalent — surface the
        # same flags the GUI's wrangler_finished handler checks so it
        # auto-reloads the generated file and selects the last frame
        # once processing is done. Without these, the display is left
        # showing stale state from before the run.
        self.batch_mode = True
        self.xye_only = False

    # ── Main entry point ─────────────────────────────────────────────────

    def run(self):
        """Read frames from NeXus file and integrate each one."""
        t0 = time.time()
        if self.poni is None or not self.nexus_file:
            return

        # Setup detector and mask
        self.detector = get_detector(self.poni.detector) if self.poni.detector else None
        det_mask = self.detector.mask if self.detector is not None else None
        if self.mask_file and os.path.exists(self.mask_file):
            custom_mask = np.asarray(read_image(self.mask_file), dtype=bool)
            det_mask = det_mask | custom_mask if det_mask is not None else custom_mask
        self.mask = np.flatnonzero(det_mask) if det_mask is not None else None

        # Sync GI mode
        if self.gi:
            self.sphere.bai_1d_args['gi_mode_1d'] = self.gi_mode_1d
            self.sphere.bai_2d_args['gi_mode_2d'] = self.gi_mode_2d

        # Find the image dataset in the NeXus file
        img_ds_path = find_nexus_image_dataset(self.nexus_file, self.entry)
        if img_ds_path is None:
            self.showLabel.emit('No image dataset found in NeXus file')
            return

        # Read metadata if available
        try:
            scan_meta = read_nexus(self.nexus_file, self.entry)
            img_meta = {}
            for k, v in scan_meta.counters.items():
                if len(v) > 0:
                    img_meta[k] = float(v[0])
            for k, v in scan_meta.angles.items():
                if len(v) > 0:
                    img_meta[k] = float(v[0])
        except Exception:
            img_meta = {}

        scan_name = Path(self.nexus_file).stem

        # Initialize sphere
        sphere = self._initialize_sphere(scan_name)
        integrator = poni_to_integrator(self.poni)
        sphere._cached_integrator = integrator
        sphere._cached_fiber_integrator = None

        # Notify of new scan
        self.sigUpdateFile.emit(
            scan_name, self.fname,
            self.gi, self.th_mtr,
            False, False,  # single_img=False, series_average=False
        )

        # Process frames
        files_processed = 0
        with h5py.File(self.nexus_file, 'r') as h5f:
            ds = h5f[img_ds_path]
            nframes = ds.shape[0]
            self.showLabel.emit(f'Found {nframes} frames in {Path(self.nexus_file).name}')

            for frame_idx in range(nframes):
                if self.command == 'stop':
                    break

                self.showLabel.emit(f'Processing frame {frame_idx + 1}/{nframes}')

                # Read frame data
                img_data = np.asarray(ds[frame_idx], dtype=float)

                # Per-frame metadata from scan arrays
                frame_meta = dict(img_meta)
                try:
                    for k, v in scan_meta.counters.items():
                        if frame_idx < len(v):
                            frame_meta[k] = float(v[frame_idx])
                    for k, v in scan_meta.angles.items():
                        if frame_idx < len(v):
                            frame_meta[k] = float(v[frame_idx])
                except Exception:
                    pass

                # Integrate
                self._process_one(sphere, frame_idx, img_data, frame_meta)
                files_processed += 1

        # Final HDF5 save
        if files_processed > 0 and self.command != 'stop':
            _get_h5pool().pause(sphere.data_file)
            try:
                with self.file_lock:
                    sphere.save_to_h5(data_only=False, replace=False)
            finally:
                _get_h5pool().resume(sphere.data_file)

        self.showLabel.emit(f'Done — {files_processed} frames processed')
        print(f'NeXus Total Time: {time.time() - t0:.2f}s, {files_processed} frames')

    def _initialize_sphere(self, scan_name):
        """Create or reset the EwaldSphere for this scan."""
        self.sphere.name = scan_name
        self.sphere.gi = self.gi
        self.sphere.static = True
        return self.sphere

    def _process_one(self, sphere, frame_idx, img_data, img_meta):
        """Integrate one frame and save results."""
        _t1 = time.time()
        arch = EwaldArch(
            frame_idx, img_data, poni=self.poni,
            scan_info=img_meta, static=True, gi=self.gi,
            th_mtr=self.th_mtr,
            sample_orientation=self.sample_orientation,
            tilt_angle=self.tilt_angle,
            series_average=False,
            integrator=sphere._cached_integrator,
        )

        if self.gi:
            _incident_angle = arch._get_incident_angle()
            if (sphere._cached_fiber_integrator is None
                    or _incident_angle != self._cached_gi_incident_angle):
                sphere._cached_fiber_integrator = create_fiber_integrator(
                    arch._poni_from_integrator(),
                    incident_angle=_incident_angle,
                    tilt_angle=arch.tilt_angle,
                    sample_orientation=self.sample_orientation,
                    angle_unit="deg",
                )
                self._cached_gi_incident_angle = _incident_angle

        arch.integrate_1d(
            global_mask=self.mask,
            fiber_integrator=sphere._cached_fiber_integrator,
            **sphere.bai_1d_args,
        )

        if not sphere.skip_2d:
            arch.integrate_2d(
                global_mask=self.mask,
                fiber_integrator=sphere._cached_fiber_integrator,
                **sphere.bai_2d_args,
            )

        # Populate in-memory data
        self.data_1d[frame_idx] = arch.copy(include_2d=False)
        self.data_2d[frame_idx] = {
            'map_raw': arch.map_raw,
            'bg_raw': arch.bg_raw,
            'mask': arch.mask,
            'int_2d': arch.int_2d,
            'gi_2d': arch.gi_2d,
            'thumbnail': None,
        }

        # Set source file reference
        try:
            arch.source_file = os.path.relpath(
                self.nexus_file, os.path.dirname(sphere.data_file))
        except ValueError:
            arch.source_file = str(self.nexus_file)
        arch.skip_map_raw = sphere.skip_2d  # NeXus frames stay in source

        # Save to HDF5
        _get_h5pool().pause(sphere.data_file)
        try:
            with self.file_lock:
                with _catch_h5(sphere.data_file, 'a') as h5f:
                    sphere.add_arch(
                        arch=arch, calculate=False, update=True,
                        get_sd=True, set_mg=False, static=True, gi=self.gi,
                        th_mtr=self.th_mtr, series_average=False,
                        h5file=h5f,
                    )
        finally:
            _get_h5pool().resume(sphere.data_file)

        _t_total = time.time() - _t1
        print(f'[NEXUS] frame_{frame_idx:04d}: total={_t_total:.2f}s')
        self.sigUpdate.emit(frame_idx)
