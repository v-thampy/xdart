# -*- coding: utf-8 -*-
"""
nexusThread — worker thread for NeXus/Tiled wrangler.

Reads frames from a NeXus HDF5 file (Bluesky suitcase-nexus format)
and integrates each one using the same LiveFrame pipeline as imageThread.

Performance shape (post-P3A refactor 2026-05-13):

* **Bulk HDF5 reads** — frames are read in ``_READ_CHUNK``-sized
  slices (``ds[a:b]``), so HDF5 chunk decompression happens once per
  N frames rather than N times for the same chunk.  Reads go through
  :class:`ssrl_xrd_tools.io.nexus.NexusImageStack`, which exposes a
  single (N, H, W) logical view across either a single 3D dataset or
  an Eiger master's sibling ``data_NNNNNN`` external links — chunked
  reads cross file boundaries seamlessly.
* **Parallel integration** — within each chunk, integration is
  dispatched to a ``ThreadPoolExecutor`` backed by a per-scan
  :class:`IntegratorPool` (one pyFAI integrator per worker — pyFAI's
  CSR engine isn't thread-safe with different inputs on a shared
  instance; see xdart.utils.integrator_pool).
* **Periodic saves** — disk writes are batched every
  ``_LIVE_SAVE_INTERVAL`` frames so the v2 NeXus writer's per-flush
  cost amortises across the scan.  Skipped entirely under
  ``xye_only`` mode (Int 1D (XYE)).
* **Per-chunk XYE flush** — XYE files are buffered inside the worker
  and flushed once per chunk by ``_flush_xye_buffer`` (inherited from
  wranglerThread).  Buffering keeps the worker thread cheap and
  groups disk traffic so it doesn't interleave with the next chunk's
  integration.  Per-frame XYE export happens in **every** mode
  (Int 1D + 2D, Int 1D, Int 1D (XYE)).
* **GI mode safe** — the fiber integrator is cached on the scan
  **before** the parallel section starts (using frame 0 to compute
  the incident angle), so workers only read it.
* **1D-only mode** — set ``scan.skip_2d = True`` to bypass 2D
  integration entirely (faster on large detectors).  Set
  ``self.xye_only = True`` (in addition) to also bypass the .nxs
  writer and produce XYE files only.

@author: thampy
"""

# Standard library imports
import logging
import os
import time
from pathlib import Path

import numpy as np

# Qt imports
from pyqtgraph import Qt

# Project imports
from xdart.modules.live import LiveFrame, IncidenceAngleUnresolved
from ssrl_xrd_tools.integrate.gid import create_fiber_integrator
from ssrl_xrd_tools.integrate.calibration import poni_to_integrator, get_detector
from ssrl_xrd_tools.io.nexus import open_nexus_image_stack, read_nexus
from ssrl_xrd_tools.io.image import read_image
from ssrl_xrd_tools.io.export import write_xye
from xdart.utils.h5pool import get_pool as _get_h5pool
from xdart.utils.integrator_pool import ensure_integrator_pool
from xdart.modules.reduction import (
    StandardPlanCache,
    dispatch_live_frame_reduction,
    sync_live_scan_gi_settings,
)
from .wrangler_widget import wranglerThread

logger = logging.getLogger(__name__)


# How many frames to bulk-read from the source HDF5 per iteration.
# HDF5 chunks for typical detectors hold a handful of frames each;
# reading in 16-frame slices avoids paying per-chunk decompression
# multiple times for the same chunk.
_READ_CHUNK = 16

# Save cadence inherited from wranglerThread.LIVE_SAVE_INTERVAL.
# Subclass-level override would go here as ``LIVE_SAVE_INTERVAL = N``.

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
            scan_args,
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
            scan,
            data_1d,
            data_2d,
            entry='entry',
            parent=None):

        super().__init__(command_queue, scan_args, fname, file_lock, parent)

        self.nexus_file = nexus_file
        self.poni = poni
        self.mask_file = mask_file
        self.gi = gi
        self.incidence_motor = th_mtr
        self.sample_orientation = sample_orientation
        self.tilt_angle = tilt_angle
        self.gi_mode_1d = gi_mode_1d
        self.gi_mode_2d = gi_mode_2d
        self.command = command
        self.scan = scan
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
        # C1: cached standard ReductionPlan, rebuilt only when scan
        # settings change.  Lives on the thread so it survives across
        # chunks within a single run.
        self._plan_cache = StandardPlanCache()

    # ── Main entry point ─────────────────────────────────────────────────

    def run(self):
        """QThread entry: run the integration body, always reclaiming the
        integration pool — even on an exception-aborted run — so worker threads
        don't outlive the run (matches imageThread.run()'s finally)."""
        try:
            self._run_impl()
        finally:
            self._shutdown_executor()

    def _run_impl(self):
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
            self.scan.bai_1d_args['gi_mode_1d'] = self.gi_mode_1d
            self.scan.bai_2d_args['gi_mode_2d'] = self.gi_mode_2d

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
        scan = self._initialize_scan(scan_name)
        scan._cached_integrator = poni_to_integrator(self.poni)
        scan._cached_fiber_integrator = None

        # Notify the GUI that a new scan is being processed.
        self.sigUpdateFile.emit(
            scan_name, self.fname,
            self.gi, self.incidence_motor,
            False, False,  # single_img=False, series_average=False
        )

        files_processed = 0
        # ``open_nexus_image_stack`` transparently handles two layouts:
        #   • single 3D dataset (e.g. /entry/instrument/detector/data)
        #   • Eiger master with sibling external links
        #     /entry/data/data_NNNNNN → individual _data_*.h5 files.
        # The proxy exposes the full scan as one (N, H, W) slice-able
        # object, so chunked reads can cross file boundaries.
        try:
            ds_cm = open_nexus_image_stack(self.nexus_file, self.entry)
        except (KeyError, FileNotFoundError) as exc:
            self.showLabel.emit(f'No image dataset found in NeXus file: {exc}')
            return
        with ds_cm as ds:
            nframes = ds.shape[0]
            n_segments = ds.n_segments
            self.showLabel.emit(
                f'Found {nframes} frames in {Path(self.nexus_file).name}'
                + (f' ({n_segments} data files)' if n_segments > 1 else '')
            )

            # ── Pre-warm GI fiber integrator BEFORE the parallel
            # section.  pyFAI's fiber integrator (like the standard
            # AzimuthalIntegrator) isn't safe to construct concurrently
            # against the shared scan attribute — and we want every
            # worker to see the same prebuilt instance.  Use frame 0
            # to compute the incident angle.
            if self.gi and scan._cached_fiber_integrator is None:
                first_frame = np.asarray(ds[0], dtype=np.float32)
                self._prewarm_fiber_integrator(scan, first_frame, base_meta)

            # F3: prewarm the stable bad-pixel mask cache on the main
            # thread before any worker runs.  Without this, the first
            # N workers all race to compute and write
            # scan._cached_data_mask (same value, but the invariant
            # isn't enforced).  Cheap: one frame read + a flatten.
            if getattr(scan, '_cached_data_mask', None) is None:
                first_frame = np.asarray(ds[0], dtype=np.float32)
                self._prewarm_frame_mask(scan, first_frame)

            n_workers = min(self.max_cores, nframes)
            # Per-scan integrator pool.  Built lazily on first parallel
            # batch; reused across all subsequent chunks so the
            # ~250 ms first-call CSR LUT cost is paid once per worker.
            integrator_pool = ensure_integrator_pool(
                scan, '_cached_integrator', n_workers,
            )
            # C1: cached per-scan plan — rebuilt only when scan
            # integration settings or mask change between chunks.
            sync_live_scan_gi_settings(
                scan,
                incidence_motor=self.incidence_motor,
                sample_orientation=self.sample_orientation,
                tilt_angle=self.tilt_angle,
            )
            standard_plan = self._plan_cache.get(
                scan, integrate_2d=not scan.skip_2d,
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
                # Shared base helper (_parallel_integrate) handles the
                # ThreadPoolExecutor + error-collection + idx-sort.
                # Each worker borrows its own integrator from the pool,
                # so pyFAI's CSR LUT cache stays valid across
                # concurrent calls.
                self.showLabel.emit(
                    f'Integrating frames {chunk_start+1}-{chunk_end}'
                    f'/{nframes} ({n_workers} workers)'
                )
                _t_phase1 = time.time()
                frames = self._parallel_integrate(
                    items,
                    lambda item: self._integrate_one(
                        scan, integrator_pool, standard_plan, *item,
                    ),
                    n_workers,
                    label='NEXUS',
                )
                _t_phase1 = time.time() - _t_phase1

                # ── Serial accumulation into the scan ─────────────
                # scan.add_frame / data_1d / data_2d mutations and
                # the sigUpdate emit happen serially; scan isn't
                # thread-safe for concurrent writes, and the GUI
                # widgets it feeds aren't either.
                for frame in frames:
                    if frame is None:
                        continue
                    self._publish(scan, frame)
                    self.sigUpdate.emit(frame.idx)
                    files_processed += 1
                    frames_since_save += 1

                # ── Per-chunk XYE flush ─────────────────────────────
                # Drain the XYE buffer once per chunk — keeps disk I/O
                # batched and prevents the buffer from growing without
                # bound on long scans.  Inherited ``_flush_xye_buffer``
                # is a no-op when the buffer is empty (e.g. on Int 2D
                # mode would be — but we always populate it).
                # P3: pass the set of frame.idx values that survived
                # ``_parallel_integrate`` so a Stop-aborted batch
                # doesn't leave orphan XYE files for frames that
                # never landed in .nxs.
                _t_xye = time.time()
                published_idxs = {a.idx for a in frames if a is not None}
                self._flush_xye_buffer(scan, published_idxs=published_idxs)
                _t_xye = time.time() - _t_xye

                logger.info(
                    '[NEXUS-BATCH] frames %d-%d  read=%.3fs  '
                    'integrate=%.3fs  xye=%.3fs  total=%.3fs',
                    chunk_start, chunk_end - 1, _t_read, _t_phase1,
                    _t_xye, _t_read + _t_phase1 + _t_xye,
                )

                # ── Periodic .nxs save ──────────────────────────────
                # ``LIVE_SAVE_INTERVAL`` (inherited from
                # wranglerThread) is checked at chunk boundaries —
                # not frame boundaries; the v2 writer's per-flush
                # cost is ~30 ms regardless, so the granularity is
                # close enough.  Skipped entirely in xye_only mode
                # (the inherited ``_save_to_disk`` is also a no-op
                # under xye_only, but we short-circuit here too so
                # the chunk loop reads clean).
                if (not self.xye_only
                        and frames_since_save >= self.LIVE_SAVE_INTERVAL):
                    self._save_to_disk(scan)
                    frames_since_save = 0

        # Final save: write everything coherent + provenance + finalize.
        #
        # N4 — Stop tail flush.  Pre-N4 this was gated on
        # ``self.command != 'stop'``, which meant that if the user
        # hit Stop after some frames had been processed and
        # published but before the next periodic save kicked in,
        # those tail frames remained in memory only and were lost.
        # Now we always do a non-finalize save on Stop so the
        # processed prefix lands on disk — only ``finalize=True``
        # (provenance + write-once items) is skipped on Stop, since
        # the scan didn't actually complete and the file should be
        # marked as a partial result.
        #
        # N7 — dropped the outer ``with self.file_lock:`` because
        # ``scan.save_to_nexus`` already takes the lock internally
        # (J2 made the locks the same Condition, so the nested
        # acquire was reentrant + redundant rather than a deadlock).
        if files_processed > 0 and not self.xye_only:
            is_finalize = (self.command != 'stop')
            _get_h5pool().pause(scan.data_file)
            try:
                scan.default_geometry()
                scan.save_to_nexus(
                    replace=False, finalize=is_finalize,
                )
            finally:
                _get_h5pool().resume(scan.data_file)
            if not is_finalize:
                logger.info(
                    '[NEXUS] Stop tail-flushed %d frames; .nxs is '
                    'a partial result (no finalize stamp).',
                    files_processed,
                )

        self.showLabel.emit(f'Done — {files_processed} frames processed')
        logger.info(
            'NeXus total time: %.2fs, %d frames', time.time() - t0,
            files_processed,
        )
        # Pool reclamation is handled by run()'s finally (covers normal AND
        # exception-aborted runs).

    # ── Helpers ─────────────────────────────────────────────────────────

    def _initialize_scan(self, scan_name):
        """Create or reset the LiveScan for this scan."""
        self.scan.name = scan_name
        self.scan.gi = self.gi
        self.scan.static = True
        return self.scan

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
        except (AttributeError, TypeError, ValueError, KeyError) as e:
            # AttributeError: scan_meta missing counters/angles attr.
            # TypeError/ValueError: counter array contains non-numeric.
            # KeyError: shouldn't happen but be defensive on dict-like.
            # Any of these falls back to base_meta — already populated.
            logger.debug(
                "frame_meta lookup failed for frame %s: %s", frame_idx, e,
            )
        return meta

    def _prewarm_fiber_integrator(self, scan, first_frame, base_meta):
        """Build ``scan._cached_fiber_integrator`` from frame 0.

        We need this *before* the parallel section starts so workers
        can read the cached instance instead of racing to create one.
        Same pattern as imageThread.
        """
        # Construct a throw-away frame just to compute the incidence
        # angle from frame 0 + base metadata.  The frame isn't kept.
        scratch = LiveFrame(
            0, first_frame, poni=self.poni,
            scan_info=base_meta, static=True, gi=self.gi,
            th_mtr=self.incidence_motor,
            sample_orientation=self.sample_orientation,
            tilt_angle=self.tilt_angle,
            series_average=False,
            integrator=scan._cached_integrator,
        )
        try:
            incident_angle = scratch._get_incident_angle()
        except IncidenceAngleUnresolved as exc:
            # No resolvable incidence — skip prewarm and surface the fix.
            # Per-frame integration raises the same way; don't seed a
            # degenerate 0° fiber integrator on the scan.
            self.showLabel.emit(
                'GI needs an incidence angle: set Theta Motor to Manual '
                'and enter the angle.'
            )
            logger.warning('GI fiber-integrator prewarm skipped: %s', exc)
            return
        scan._cached_fiber_integrator = create_fiber_integrator(
            scratch._poni_from_integrator(),
            incident_angle=incident_angle,
            tilt_angle=scratch.tilt_angle,
            sample_orientation=self.sample_orientation,
            angle_unit="deg",
        )
        # Cache the angle so the parallel workers in :meth:`_integrate_one`
        # can detect drift on ω-varying scans and fall back to a
        # worker-local fiber integrator at the correct angle.  Same
        # pattern as imageThread.
        scan._cached_fiber_integrator_angle = incident_angle

    def _integrate_one(self, scan, integrator_pool, standard_plan,
                       frame_idx, img_data, img_meta):
        """Pure integration in a worker thread; returns the frame.

        Borrows an integrator from the pool, integrates 1D/2D, and
        detaches the pool's instance from the frame before returning
        so the next worker can borrow it safely.  All shared-scan
        mutation (data_1d / data_2d / add_frame / sigUpdate) happens
        on the main thread in :meth:`_publish`, not here.

        F2 cancel-fast: returns ``None`` immediately when Stop has
        been requested, so the user doesn't wait for pyFAI to
        finish the current frame after pressing Stop.
        """
        if self.command == 'stop':
            return None
        _t0 = time.time()
        # Stable bad-pixel mask cached on the scan across frames so
        # pyFAI's CSR cache stays valid.  Helper now lives on the
        # wranglerThread base class — same impl as imageThread uses.
        frame_mask = self._resolve_frame_mask(scan, img_data)

        # Borrow a private integrator — pyFAI isn't thread-safe with
        # different inputs on a shared instance.
        with integrator_pool.borrow() as ai:
            frame = LiveFrame(
                frame_idx, img_data, poni=self.poni,
                scan_info=img_meta, static=True, gi=self.gi,
                th_mtr=self.incidence_motor,
                sample_orientation=self.sample_orientation,
                tilt_angle=self.tilt_angle,
                series_average=False,
                integrator=ai,
                mask=frame_mask,
            )

            dispatch_live_frame_reduction(
                frame, scan,
                standard_plan=standard_plan,
                integrator=ai,
                global_mask=self.mask,
            )

        # Detach the pool integrator from the frame — once the `with`
        # exits, another worker can borrow this same instance.  The
        # scan's source integrator is safe to share with non-pool
        # consumers (GUI display, etc.).
        frame.integrator = scan._cached_integrator

        # Set source file reference for the v2 NeXus per-frame group.
        # The source_frame_idx is the *global* frame index across all
        # external-link data files (matches NexusImageStack's flattened
        # view), so a lazy raw loader can do
        # ``NexusImageStack(source_file)[source_frame_idx]`` directly
        # without needing to know which data_NNNNNN segment to open.
        frame.source_file = os.path.abspath(str(self.nexus_file))
        frame.source_frame_idx = int(frame_idx)
        # NeXus frames already live in the source — don't double-store
        # them in the output .nxs.
        frame.skip_map_raw = True

        # Buffer the XYE write — flushed at end of each chunk by the
        # main loop.  Mirrors the imageThread pattern; keeps the worker
        # thread cheap and groups disk traffic so it doesn't interleave
        # with the next chunk's integration.  The flush itself respects
        # ``xye_only`` mode (see ``_flush_xye_buffer``); the buffer is
        # populated unconditionally because we always want per-frame
        # XYE files in Int 1D / Int 1D + 2D / Int 1D (XYE) modes.
        with self._xye_lock:
            self._xye_buffer.append((frame_idx, frame))

        logger.debug(
            '[NEXUS] frame_%04d integrated in %.3fs',
            frame_idx, time.time() - _t0,
        )
        return frame

    def _publish(self, scan, frame):
        """Push the integrated frame into scan + the publish slot.

        Runs on the main thread after the parallel section so
        ``scan.add_frame`` is serialised — scan isn't thread-safe
        for concurrent writes.

        After D1 (unified handoff): we no longer write
        ``self.data_1d``/``self.data_2d`` directly.  Instead we leave
        the frame in ``self._published_frames[frame.idx]`` and emit
        ``sigUpdate``; ``static_scan_widget.update_data`` pops it and
        owns the dict updates + bounded eviction.  Same single
        source-of-truth contract that imageThread's live path uses.
        """
        # In-memory accumulate only — the chunked flush at the end of
        # the dispatch loop (and the final ``save_to_nexus(finalize=True)``
        # at the bottom of ``run()``) handle persistence.
        scan.add_frame(
            frame=frame, calculate=False, update=True,
            get_sd=True, set_mg=False, static=True, gi=self.gi,
            th_mtr=self.incidence_motor, series_average=False,
            batch_save=True,
        )
        # Publish for the GUI's update_data slot to consume.  Single
        # write site for the dict round-trip.
        self._published_frames[frame.idx] = frame

    # ``_save_to_disk`` is inherited from wranglerThread.  Called
    # from the chunk loop every LIVE_SAVE_INTERVAL frames so the
    # on-disk file stays close to in-memory state even if the user
    # kills the process mid-scan.
