# -*- coding: utf-8 -*-
"""
nexusThread — worker thread for NeXus/Tiled wrangler.

Reads frames from a NeXus HDF5 file (Bluesky suitcase-nexus format)
and integrates each one using the same EwaldArch pipeline as specThread.

Performance shape (post-P3A refactor 2026-05-13):

* **Bulk HDF5 reads** — frames are read in ``_READ_CHUNK``-sized
  slices (``ds[a:b]``), so HDF5 chunk decompression happens once per
  N frames rather than N times for the same chunk.
* **Parallel integration** — within each chunk, integration is
  dispatched to a ``ThreadPoolExecutor`` backed by a per-scan
  :class:`IntegratorPool` (one pyFAI integrator per worker — pyFAI's
  CSR engine isn't thread-safe with different inputs on a shared
  instance; see xdart.utils.integrator_pool).
* **Periodic saves** — disk writes are batched every
  ``_LIVE_SAVE_INTERVAL`` frames so the v2 NeXus writer's per-flush
  cost amortises across the scan.
* **GI mode safe** — the fiber integrator is cached on the sphere
  **before** the parallel section starts (using frame 0 to compute
  the incident angle), so workers only read it.

@author: thampy
"""

# Standard library imports
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

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
from xdart.utils.h5pool import get_pool as _get_h5pool
from xdart.utils.integrator_pool import ensure_integrator_pool
from .wrangler_widget import wranglerThread

logger = logging.getLogger(__name__)


# How many frames to bulk-read from the source HDF5 per iteration.
# HDF5 chunks for typical detectors hold a handful of frames each;
# reading in 16-frame slices avoids paying per-chunk decompression
# multiple times for the same chunk.
_READ_CHUNK = 16

# How many frames between disk saves in the dispatch loop.  The v2
# writer's per-flush cost is ~30 ms regardless of total scan length
# (incremental append-only), so 8 is a comfortable cadence that
# bounds frame-loss-on-crash to ~8 frames without dominating
# the wall-clock.
_LIVE_SAVE_INTERVAL = 8

# Default number of parallel integration workers when the GUI
# doesn't expose a Cores spinbox to the NeXus wrangler (it currently
# doesn't — wiring it up is a small UI follow-up).  Caller can set
# ``self.max_cores`` before starting the thread to override.
_DEFAULT_MAX_CORES = 4


class nexusThread(wranglerThread):
    """Thread for processing NeXus/HDF5 image stacks.

    Reads an image dataset from a NeXus file, integrates each frame
    via a parallel ``ThreadPoolExecutor``, and emits ``sigUpdate``
    after each one.

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

        # NeXus processing is always batch-mode equivalent — surface the
        # same flags the GUI's wrangler_finished handler checks so it
        # auto-reloads the generated file and selects the last frame
        # once processing is done. Without these, the display is left
        # showing stale state from before the run.
        self.batch_mode = True
        self.xye_only = False
        # Settable from outside (e.g. before .start()) when the GUI
        # eventually exposes a Cores spinbox for the NeXus wrangler.
        self.max_cores = _DEFAULT_MAX_CORES

    # ── Main entry point ─────────────────────────────────────────────────

    def run(self):
        """Read frames from a NeXus file and integrate them in parallel."""
        t0 = time.time()
        if self.poni is None or not self.nexus_file:
            return

        # Setup detector and global mask
        self.detector = (get_detector(self.poni.detector)
                         if self.poni.detector else None)
        det_mask = self.detector.mask if self.detector is not None else None
        if self.mask_file and os.path.exists(self.mask_file):
            custom_mask = np.asarray(read_image(self.mask_file), dtype=bool)
            det_mask = (det_mask | custom_mask if det_mask is not None
                        else custom_mask)
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

        # Read scan-level metadata once (counters/angles per-frame
        # arrays).  Per-frame slicing happens later.
        try:
            scan_meta = read_nexus(self.nexus_file, self.entry)
            base_meta = {}
            for k, v in scan_meta.counters.items():
                if len(v) > 0:
                    base_meta[k] = float(v[0])
            for k, v in scan_meta.angles.items():
                if len(v) > 0:
                    base_meta[k] = float(v[0])
        except Exception:
            scan_meta = None
            base_meta = {}

        scan_name = Path(self.nexus_file).stem
        sphere = self._initialize_sphere(scan_name)
        sphere._cached_integrator = poni_to_integrator(self.poni)
        sphere._cached_fiber_integrator = None

        # Notify the GUI that a new scan is being processed.
        self.sigUpdateFile.emit(
            scan_name, self.fname,
            self.gi, self.th_mtr,
            False, False,  # single_img=False, series_average=False
        )

        files_processed = 0
        with h5py.File(self.nexus_file, 'r') as h5f:
            ds = h5f[img_ds_path]
            nframes = ds.shape[0]
            self.showLabel.emit(
                f'Found {nframes} frames in {Path(self.nexus_file).name}'
            )

            # ── Pre-warm GI fiber integrator BEFORE the parallel
            # section.  pyFAI's fiber integrator (like the standard
            # AzimuthalIntegrator) isn't safe to construct concurrently
            # against the shared sphere attribute — and we want every
            # worker to see the same prebuilt instance.  Use frame 0
            # to compute the incident angle.
            if self.gi and sphere._cached_fiber_integrator is None:
                first_frame = np.asarray(ds[0], dtype=np.float32)
                self._prewarm_fiber_integrator(sphere, first_frame, base_meta)

            n_workers = min(self.max_cores, nframes)
            # Per-scan integrator pool.  Built lazily on first parallel
            # batch; reused across all subsequent chunks so the
            # ~250 ms first-call CSR LUT cost is paid once per worker.
            integrator_pool = ensure_integrator_pool(
                sphere, '_cached_integrator', n_workers,
            )
            # If the GUI is single-core (max_cores=1) integrator_pool
            # still exists with one member; the worker borrows it like
            # everyone else.  Cleaner than branching on n_workers==1.

            frames_since_save = 0
            for chunk_start in range(0, nframes, _READ_CHUNK):
                if self.command == 'stop':
                    break
                chunk_end = min(chunk_start + _READ_CHUNK, nframes)
                chunk_size = chunk_end - chunk_start

                # Bulk-read the chunk — one HDF5 decompression pass.
                _t_read = time.time()
                block = np.asarray(ds[chunk_start:chunk_end],
                                   dtype=np.float32)
                _t_read = time.time() - _t_read

                # Build per-frame work items.
                items = []
                for i, frame_idx in enumerate(range(chunk_start, chunk_end)):
                    items.append((
                        frame_idx,
                        block[i],
                        self._frame_meta(scan_meta, base_meta, frame_idx),
                    ))

                # ── Parallel integration ────────────────────────────
                # ThreadPoolExecutor + integrator pool: each worker
                # borrows its own integrator copy, so the CSR LUT
                # cache stays valid across concurrent calls.
                self.showLabel.emit(
                    f'Integrating frames {chunk_start+1}-{chunk_end}'
                    f'/{nframes} ({n_workers} workers)'
                )
                _t_phase1 = time.time()
                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    arches = list(pool.map(
                        lambda item: self._integrate_one(
                            sphere, integrator_pool, *item,
                        ),
                        items,
                    ))
                _t_phase1 = time.time() - _t_phase1

                # ── Serial accumulation into the sphere ─────────────
                # sphere.add_arch / data_1d / data_2d mutations and
                # the sigUpdate emit happen serially; sphere isn't
                # thread-safe for concurrent writes, and the GUI
                # widgets it feeds aren't either.
                for arch in arches:
                    if arch is None:
                        continue
                    self._publish(sphere, arch)
                    self.sigUpdate.emit(arch.idx)
                    files_processed += 1
                    frames_since_save += 1

                logger.info(
                    '[NEXUS-BATCH] frames %d-%d  read=%.3fs  '
                    'integrate=%.3fs  total=%.3fs',
                    chunk_start, chunk_end - 1, _t_read, _t_phase1,
                    _t_read + _t_phase1,
                )

                # ── Periodic save ───────────────────────────────────
                # _LIVE_SAVE_INTERVAL is checked at chunk boundaries
                # (not frame boundaries); the v2 writer's per-flush
                # cost is ~30 ms regardless, so the granularity is
                # close enough.
                if frames_since_save >= _LIVE_SAVE_INTERVAL:
                    self._flush_to_disk(sphere)
                    frames_since_save = 0

        # Final save: write everything coherent + provenance + finalize
        # (last call of the scan).
        if files_processed > 0 and self.command != 'stop':
            _get_h5pool().pause(sphere.data_file)
            try:
                with self.file_lock:
                    sphere.default_geometry()
                    sphere.save_to_nexus(replace=False, finalize=True)
            finally:
                _get_h5pool().resume(sphere.data_file)

        self.showLabel.emit(f'Done — {files_processed} frames processed')
        logger.info(
            'NeXus total time: %.2fs, %d frames', time.time() - t0,
            files_processed,
        )

    # ── Helpers ─────────────────────────────────────────────────────────

    def _initialize_sphere(self, scan_name):
        """Create or reset the EwaldSphere for this scan."""
        self.sphere.name = scan_name
        self.sphere.gi = self.gi
        self.sphere.static = True
        return self.sphere

    def _frame_meta(self, scan_meta, base_meta, frame_idx):
        """Build a per-frame metadata dict from scan-level arrays.

        Falls back to ``base_meta`` (frame 0's slice) when the scan
        arrays are shorter than expected — keeps the call cheap and
        avoids exceptions in the parallel section.
        """
        meta = dict(base_meta)
        if scan_meta is None:
            return meta
        try:
            for k, v in scan_meta.counters.items():
                if frame_idx < len(v):
                    meta[k] = float(v[frame_idx])
            for k, v in scan_meta.angles.items():
                if frame_idx < len(v):
                    meta[k] = float(v[frame_idx])
        except Exception:
            pass
        return meta

    def _prewarm_fiber_integrator(self, sphere, first_frame, base_meta):
        """Build ``sphere._cached_fiber_integrator`` from frame 0.

        We need this *before* the parallel section starts so workers
        can read the cached instance instead of racing to create one.
        Same pattern as specThread.
        """
        # Construct a throw-away arch just to compute the incidence
        # angle from frame 0 + base metadata.  The arch isn't kept.
        scratch = EwaldArch(
            0, first_frame, poni=self.poni,
            scan_info=base_meta, static=True, gi=self.gi,
            th_mtr=self.th_mtr,
            sample_orientation=self.sample_orientation,
            tilt_angle=self.tilt_angle,
            series_average=False,
            integrator=sphere._cached_integrator,
        )
        incident_angle = scratch._get_incident_angle()
        sphere._cached_fiber_integrator = create_fiber_integrator(
            scratch._poni_from_integrator(),
            incident_angle=incident_angle,
            tilt_angle=scratch.tilt_angle,
            sample_orientation=self.sample_orientation,
            angle_unit="deg",
        )

    def _integrate_one(self, sphere, integrator_pool, frame_idx,
                       img_data, img_meta):
        """Pure integration in a worker thread; returns the arch.

        Borrows an integrator from the pool, integrates 1D/2D, and
        detaches the pool's instance from the arch before returning
        so the next worker can borrow it safely.  All shared-sphere
        mutation (data_1d / data_2d / add_arch / sigUpdate) happens
        on the main thread in :meth:`_publish`, not here.
        """
        _t0 = time.time()
        # Stable bad-pixel mask cached on the sphere across frames so
        # pyFAI's CSR cache stays valid — same trick specThread uses.
        if getattr(sphere, '_cached_data_mask', None) is None:
            try:
                sphere._cached_data_mask = np.arange(img_data.size)[
                    np.asarray(img_data).flatten() < 0
                ]
            except Exception:
                sphere._cached_data_mask = None

        # Borrow a private integrator — pyFAI isn't thread-safe with
        # different inputs on a shared instance.
        with integrator_pool.borrow() as ai:
            arch = EwaldArch(
                frame_idx, img_data, poni=self.poni,
                scan_info=img_meta, static=True, gi=self.gi,
                th_mtr=self.th_mtr,
                sample_orientation=self.sample_orientation,
                tilt_angle=self.tilt_angle,
                series_average=False,
                integrator=ai,
                mask=sphere._cached_data_mask,
            )

            # GI fiber integrator: pre-warmed on the main thread (see
            # _prewarm_fiber_integrator); workers only read it.
            fi = sphere._cached_fiber_integrator

            arch.integrate_1d(
                global_mask=self.mask,
                fiber_integrator=fi,
                **sphere.bai_1d_args,
            )
            if not sphere.skip_2d:
                arch.integrate_2d(
                    global_mask=self.mask,
                    fiber_integrator=fi,
                    **sphere.bai_2d_args,
                )

        # Detach the pool integrator from the arch — once the `with`
        # exits, another worker can borrow this same instance.  The
        # sphere's source integrator is safe to share with non-pool
        # consumers (GUI display, etc.).
        arch.integrator = sphere._cached_integrator

        # Set source file reference for the v2 NeXus per-frame group.
        try:
            arch.source_file = os.path.relpath(
                self.nexus_file, os.path.dirname(sphere.data_file),
            )
        except ValueError:
            arch.source_file = str(self.nexus_file)
        # NeXus frames already live in the source — don't double-store
        # them in the output .nxs.
        arch.skip_map_raw = True

        logger.debug(
            '[NEXUS] frame_%04d integrated in %.3fs',
            frame_idx, time.time() - _t0,
        )
        return arch

    def _publish(self, sphere, arch):
        """Push the integrated arch into sphere + GUI display state.

        Runs on the main thread after the parallel section so
        ``sphere.add_arch`` and the per-frame data dict updates are
        serialised — sphere isn't thread-safe for concurrent writes.
        """
        self.data_1d[arch.idx] = arch.copy(include_2d=False)
        self.data_2d[arch.idx] = {
            'map_raw': arch.map_raw,
            'bg_raw': arch.bg_raw,
            'mask': arch.mask,
            'int_2d': arch.int_2d,
            'gi_2d': arch.gi_2d,
            'thumbnail': None,
        }
        # In-memory accumulate only — the chunked flush at the end of
        # the dispatch loop (and the final ``save_to_nexus(finalize=True)``
        # at the bottom of ``run()``) handle persistence.
        sphere.add_arch(
            arch=arch, calculate=False, update=True,
            get_sd=True, set_mg=False, static=True, gi=self.gi,
            th_mtr=self.th_mtr, series_average=False,
            batch_save=True,
        )

    def _flush_to_disk(self, sphere):
        """Save accumulated sphere state to disk (intermediate save).

        Called from the chunk loop every ``_LIVE_SAVE_INTERVAL`` frames
        so the on-disk file stays close to in-memory state even if the
        user kills the process mid-scan.
        """
        _get_h5pool().pause(sphere.data_file)
        try:
            with self.file_lock:
                sphere._save_to_nexus()
        finally:
            _get_h5pool().resume(sphere.data_file)
