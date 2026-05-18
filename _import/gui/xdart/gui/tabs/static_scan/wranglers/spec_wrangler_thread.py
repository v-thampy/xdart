# -*- coding: utf-8 -*-
"""
specThread — worker thread for spec_wrangler.

Handles all image processing, integration, background subtraction,
and file I/O in a separate QThread.

@author: thampy, walroth
"""

# Standard library imports
import logging
import os
import queue
import re
import threading
import time
import glob
import numpy as np
from pathlib import Path
from collections import deque

# Per-worker pyFAI integrator pool — required for correct parallel
# batch mode.  pyFAI's AzimuthalIntegrator isn't thread-safe across
# different inputs on a shared instance, so each worker borrows its
# own integrator copy from a per-scan pool.  See module docstring of
# xdart.utils.integrator_pool for the full story.
from xdart.utils.integrator_pool import ensure_integrator_pool

logger = logging.getLogger(__name__)

# pyFAI / fabio / h5py
import fabio
import h5py

# Qt imports
from pyqtgraph import Qt

# Project imports
from xdart.modules.ewald import EwaldArch, EwaldSphere
from ssrl_xrd_tools.integrate.gid import create_fiber_integrator
from ssrl_xrd_tools.integrate.calibration import poni_to_integrator, get_detector
from ssrl_xrd_tools.io.image import read_image, count_frames
from ssrl_xrd_tools.io.export import write_xye
from ssrl_xrd_tools.io.nexus import find_nexus_image_dataset
from ssrl_xrd_tools.io.metadata import read_image_metadata
from xdart.utils import get_series_avg
from xdart.utils.h5pool import get_pool as _get_h5pool
from xdart.modules.reduction import (
    StandardPlanCache,
    dispatch_arch_reduction,
)
from .wrangler_widget import wranglerThread


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _is_eiger_master(path):
    """Return True if path looks like an Eiger HDF5 master file (*_master.h5 / *_master.hdf5)."""
    return Path(path).stem.endswith('_master')


def _raw_lives_in_source(path):
    """Return True if the raw image for ``path`` is already embedded in the
    source file (Eiger master or any HDF5/NeXus container).

    When this is the case, writing ``map_raw`` into the output sphere HDF5
    is pure duplication and can dominate the per-frame write cost.
    """
    if not path:
        return False
    ext = Path(path).suffix.lower()
    return ext in _RAW_EMBEDDED_EXTS or _is_eiger_master(path)


def _get_scan_info(fname):
    """Return (scan_name, img_number) for a file path.

    Strips trailing _<digits> suffix from the stem to get scan_name.
    Falls back to (stem, None) when no numeric suffix is found.
    """
    stem = Path(fname).stem
    try:
        img_number = int(stem[stem.rindex('_') + 1:])
        scan_name = stem[:stem.rindex('_')]
    except ValueError:
        scan_name = stem
        img_number = None
    return scan_name, img_number


# ---------------------------------------------------------------------------
# Natural sort helpers
# ---------------------------------------------------------------------------

# Pre-compiled regex patterns (avoids recompilation on every sort key call)
_INT_PATTERN = re.compile(r'(\d+)')
_FLOAT_PATTERN = re.compile(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)')

# Maximum number of frames to accumulate in the pending list before
# dispatching a partial batch.  Keeps peak memory bounded when the input
# contains thousands of frames in a single file (e.g. Bluesky .nxs with
# a 1000-frame Eiger stack at 2167x2070 int32 ~= 18 GB if collected in
# one go).  Larger values amortise ThreadPoolExecutor startup and the
# end-of-batch sphere._save_to_nexus flush across more frames, so raise
# this when RAM allows.
#
# 16 frames of a 2167x2070 int32 Eiger image is ~300 MB of peak buffer
# — fits comfortably on a slow-disk laptop while still giving the user
# visible per-batch UI refreshes every ~25 s at typical 1.5 s/frame
# integration.  Earlier value of 64 made first-batch latency painful on
# spinning-disk source files (16-frame reads finish ~4× sooner).
_PENDING_FLUSH_SIZE = 16

# How many integrated frames to accumulate between v2 _save_to_nexus
# Live-save cadence and threshold-sentinel constants moved to the
# wranglerThread base class (wrangler_widget.py) in May 2026 — refer
# via ``self.LIVE_SAVE_INTERVAL`` / the base's module-level
# ``_THRESHOLD_NAN`` (re-imported here for back-compat with old SPEC
# wrangler scans that may pickle/unpickle constants by name).

# Number of frames the background prefetch worker is allowed to read
# ahead of the main collect loop.  Kept small: too large and the prefetcher
# contends with the h5py writer for disk I/O (source .nxs and output HDF5
# usually share a spindle), and 18 MB/frame stacks up memory pressure that
# slows the write phase.  The main speedup for reads comes from the bulk
# read path below, not from growing this queue.
_PREFETCH_QUEUE_SIZE = 4

# How many frames the prefetcher reads from HDF5 in a single slice.  HDF5
# chunks are typically sized so that one decompression produces multiple
# frames, so reading N frames as `dset[i:i+N]` decompresses each chunk once
# instead of N times.  Frames are then dispatched one-by-one to the
# consumer queue, so downstream consumers are unaffected.
_PREFETCH_READ_CHUNK = 16

# File extensions whose raw image data already lives in the source file
# — no need to duplicate `map_raw` into the output sphere HDF5.  This is
# the single biggest write-time win for multi-frame NeXus / HDF5 inputs.
_RAW_EMBEDDED_EXTS = frozenset({'.h5', '.hdf5', '.nxs'})


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys_int(text):
    """Sort key for human-order sorting of strings with integers.

    See: http://nedbatchelder.com/blog/200712/human_sorting.html
    """
    return [atoi(c) for c in _INT_PATTERN.split(text)]


def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval


def natural_keys_float(text):
    """Sort key for human-order sorting of strings with floats.

    See: https://stackoverflow.com/a/12643073/190597
    """
    return [atof(c) for c in _FLOAT_PATTERN.split(text)]


def natural_sort_ints(list_to_sort):
    return sorted(list_to_sort, key=natural_keys_int)


def natural_sort_float(list_to_sort):
    return sorted(list_to_sort, key=natural_keys_float)


# ---------------------------------------------------------------------------
# specThread
# ---------------------------------------------------------------------------

class specThread(wranglerThread):
    """Thread for controlling image processing.  Receives and manages a
    command and signal queue to pass commands from the main thread and
    communicate back relevant signals.

    attributes:
        command_q: mp.Queue, queue to send commands to process
        file_lock: mp.Condition, process safe lock for file access
        scan_name: str, name of current scan
        fname: str, full path to data file.
        h5_dir: str, data file directory.
        img_file: str, path to image file
        img_dir: str, path to image directory
        img_ext : str, extension of image file
        series_average : bool, flag to average over series
        meta_ext : str, extension of metadata file
        poni_dict: str, Poni File name
        detector: str, Detector name
        input_q: mp.Queue, queue for commands sent from parent
        signal_q: mp.Queue, queue for commands sent from process
        sphere_args: dict, used as **kwargs in sphere initialization.
            see EwaldSphere.
        timeout: float or int, how long to continue checking for new
            data.
        command: command passed to start, stop etc.
        data_1d/2d: Dictionaries to store processed data for plotting

    signals:
        showLabel: str, sends out text to be used in specLabel

    methods:
        run: Main method, called by start
    """
    showLabel = Qt.QtCore.Signal(str)

    def __init__(
            self,
            command_queue,
            sphere_args,
            file_lock,
            fname,
            h5_dir,
            scan_name,
            single_img,
            poni,
            inp_type,
            img_file,
            img_dir,
            include_subdir,
            img_ext,
            series_average,
            meta_ext,
            file_filter,
            mask_file,
            write_mode,
            bg_type,
            bg_file,
            bg_dir,
            bg_matching_par,
            bg_match_fname,
            bg_file_filter,
            bg_scale,
            bg_norm_channel,
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
            live_mode=False,
            max_cores=1,
            parent=None):

        super().__init__(command_queue, sphere_args, fname, file_lock, parent)

        self.h5_dir = h5_dir
        self.scan_name = scan_name
        self.single_img = single_img
        self.poni = poni
        self.inp_type = inp_type
        self.img_file = img_file
        self.img_dir = img_dir
        self.include_subdir = include_subdir
        self.img_ext = img_ext
        self.series_average = series_average
        self.meta_ext = meta_ext
        self.file_filter = file_filter
        self.mask_file = mask_file
        self.write_mode = write_mode
        self.bg_type = bg_type
        self.bg_file = bg_file
        self.bg_dir = bg_dir
        self.bg_matching_par = bg_matching_par
        self.bg_match_fname = bg_match_fname
        self.bg_file_filter = bg_file_filter
        self.bg_scale = bg_scale
        self.bg_norm_channel = bg_norm_channel
        self.gi = gi
        self.incidence_motor = th_mtr
        self.sample_orientation = sample_orientation
        self.tilt_angle = tilt_angle
        self.gi_mode_1d = gi_mode_1d
        self.gi_mode_2d = gi_mode_2d
        self.live_mode = live_mode
        # batch_mode / xye_only / apply_threshold / threshold_min /
        # threshold_max / sub_label / _xye_buffer / _xye_lock /
        # _frames_since_save / _published_arches are all initialised
        # by the wranglerThread base class — see wrangler_widget.py.
        # specThread only needs to set the spec-specific extras here.
        self.max_cores = max_cores
        self.command = command
        self.sphere = sphere
        self.data_1d = data_1d
        self.data_2d = data_2d

        self.user = None
        self.mask = None
        self.detector = None
        self.img_fnames = []
        self.processed = []
        self.processed_scans = []
        # Eiger HDF5 lazy frame state
        self._eiger_master_path = None
        self._eiger_frame_idx = 0
        self._eiger_nframes = 0
        self._eiger_master_queue = deque()
        self._eiger_done_masters = set()
        self._eiger_h5_handle = None     # persistent h5py.File for Eiger reads
        self._eiger_h5_dataset = None    # dataset reference inside the open file
        self._eiger_fabio_handle = None  # persistent fabio.EigerImage fallback
        # Background prefetch state (populated on demand)
        self._prefetch_queue = None      # queue.Queue of frame tuples or None sentinel
        self._prefetch_thread = None     # threading.Thread running the reader
        self._prefetch_stop_evt = None   # threading.Event — set to cancel worker
        # O8: distinguish clean end-of-stream from worker failure.
        # ``_prefetch_worker`` sets this to the str(exception) when
        # it dies on an unexpected error; the consumer surfaces it
        # via ``showLabel`` so the user sees "Eiger read failed: ..."
        # instead of a silent end-of-scan.  None means "no error".
        self._prefetch_error = None
        self._plan_cache = StandardPlanCache()

    # ── Display helpers ──────────────────────────────────────────────────

    @staticmethod
    def _middle_truncate(text, max_len=40, ellipsis='...'):
        """Shorten ``text`` to at most ``max_len`` chars by elliding the middle.

        ``"Combi4_Angledependence_samz_4p9_03271002_0001"`` →
        ``"Combi4_Angledep..._4p9_03271002_0001"``

        Keeps the head and tail roughly balanced so the most identifying parts
        of long filenames (prefix + frame number suffix) stay visible.
        """
        if text is None:
            return ''
        if len(text) <= max_len:
            return text
        keep = max_len - len(ellipsis)
        if keep <= 0:
            return ellipsis[:max_len]
        head = (keep + 1) // 2  # bias the head slightly longer on odd splits
        tail = keep - head
        return f'{text[:head]}{ellipsis}{text[-tail:]}' if tail else f'{text[:head]}{ellipsis}'

    # ── Main entry point ─────────────────────────────────────────────────

    def run(self):
        """Initializes specProcess and watches for new commands from
        parent or signals from the process.
        """
        t0 = time.time()
        if self.poni is None or (self.img_file == ''):
            return

        self.img_fnames.clear()
        self.processed.clear()
        self.processed_scans.clear()
        self._eiger_master_path = None
        self._eiger_frame_idx = 0
        self._eiger_nframes = 0
        self._eiger_master_queue.clear()
        self._eiger_done_masters.clear()
        self._prefetch_stop_prior()     # tear down any lingering prefetcher
        self._prefetch_queue = None
        self._prefetch_thread = None
        self._prefetch_stop_evt = None
        self._prefetch_error = None
        self.detector = get_detector(self.poni.detector) if self.poni.detector else None
        self.sub_label = ''
        det_mask = self.detector.mask if self.detector is not None else None  # pyFAI .mask property
        if self.mask_file and os.path.exists(self.mask_file):
            try:
                custom_mask = np.asarray(read_image(self.mask_file), dtype=bool)
                if det_mask is not None and custom_mask.shape != det_mask.shape:
                    logger.warning('Mask file shape %s does not match detector mask shape %s — ignoring custom mask',
                                  custom_mask.shape, det_mask.shape)
                else:
                    det_mask = det_mask | custom_mask if det_mask is not None else custom_mask
            except Exception as e:
                logger.warning('Could not load mask file %s: %s', self.mask_file, e)
        self.mask = np.flatnonzero(det_mask) if det_mask is not None else None
        self._cached_gi_incident_angle = None

        # Sync GI mode selections from spec_wrangler into sphere.bai_*_args
        if self.gi:
            self.sphere.bai_1d_args['gi_mode_1d'] = self.gi_mode_1d
            self.sphere.bai_2d_args['gi_mode_2d'] = self.gi_mode_2d

        try:
            self.process_scan()
        finally:
            # Stop background prefetcher from the main thread BEFORE closing the
            # master handle.  _eiger_close_master() is called from inside the
            # prefetch worker when switching masters, so triggering a stop from
            # there would self-join — only the main thread should tear down the
            # prefetcher.
            self._prefetch_stop_prior()
            self._eiger_close_master()  # ensure Eiger handle is released
        logger.info('Total Time: %.2fs', time.time() - t0)
        # Final echo so the user can copy the output path straight from
        # the terminal without scrolling back to the per-scan banner.
        # Trailing newline gives a visual gap before the next scan's
        # 'New Scan' banner (or before the next prompt if this was the
        # last scan in the session).
        output_path = getattr(self, 'fname', None) or getattr(self.sphere, 'data_file', None)
        if output_path:
            logger.info('Output file: %s\n', output_path)

    def process_scan(self):
        """Batch-integrate all existing images, then optionally watch for new ones (live mode).

        Phase 1 — Collect: drain the current directory glob into a pending list.
        Phase 2 — Process: sequential with cached AzimuthalIntegrator (~0.35 s/frame).
        Phase 3 — Watch (live mode only): poll every 2 s for new files; process each immediately.
        """
        sphere = None
        files_processed = 0
        _cached_poni = None
        is_eiger = _is_eiger_master(self.img_file) if self.img_file else False

        # ── Phase 1 & 2: collect then process all existing images ─────────────
        pending = []  # [(img_file, img_number, img_data, img_meta, bg_raw)]
        # Per-flush read-time accumulator.  With the prefetcher this is mostly
        # queue-wait time on the main thread, not raw h5py I/O.
        _t_read_accum = 0.0

        while True:
            if self.command == 'stop':
                break

            _t_r0 = time.time()
            img_file, scan_name, img_number, img_data, img_meta = self.get_next_image()
            _t_read_this = time.time() - _t_r0
            _t_read_accum += _t_read_this
            if img_data is None:
                break  # initial glob exhausted — move on to processing
            if img_file is not None:
                fname = os.path.splitext(os.path.basename(img_file))[0]
                # When the input is a multi-frame container (HDF5/NeXus/Eiger master),
                # include the frame index so progress is visible.
                _ext = Path(img_file).suffix.lower()
                if _ext in ('.h5', '.hdf5', '.nxs') or _is_eiger_master(img_file):
                    self.showLabel.emit(
                        f'Collecting {self._middle_truncate(fname)} [frame {img_number}]'
                    )
                else:
                    self.showLabel.emit(f'Collecting {self._middle_truncate(fname)}')
            else:
                logger.warning('Invalid image file, skipping')
                continue

            img_number = 1 if img_number is None else img_number
            self.scan_name = scan_name

            # Flush and switch sphere when scan name changes
            if (sphere is None) or (scan_name != sphere.name):
                if pending:
                    files_processed += self._dispatch_batch(sphere, pending)
                    pending = []
                sphere = self.initialize_sphere()
                _cached_poni = None

            # Rebuild cached AzimuthalIntegrator when poni identity changes
            if self.poni is not _cached_poni:
                sphere._cached_integrator = poni_to_integrator(self.poni)
                sphere._cached_fiber_integrator = None
                _cached_poni = self.poni
                self._cached_gi_incident_angle = None

            if img_number in sphere.arches.index:
                if self.single_img and not is_eiger:
                    self.sigUpdate.emit(img_number)
                    break
                continue

            bg_raw = self.get_background(img_file, img_number, img_meta)
            # Stash the per-frame read time on the tuple so the per-frame
            # TIMING log can show it (otherwise it gets lumped into the
            # batch [FLUSH] line and hides per-frame variance).
            pending.append((img_file, img_number, img_data, img_meta,
                            bg_raw, _t_read_this))

            if self.single_img and not is_eiger:
                break

            # ── Dispatch cadence ────────────────────────────────────────────
            # Batch mode keeps the 16-frame pending buffer so the
            # ThreadPoolExecutor can integrate them in parallel.
            # Live (non-batch) mode dispatches immediately so the GUI
            # sees per-frame updates as soon as each image lands; the
            # disk save still gets batched separately inside
            # _dispatch_batch_serial via _frames_since_save.
            flush_size = _PENDING_FLUSH_SIZE if self.batch_mode else 1
            if len(pending) >= flush_size:
                if self.batch_mode:
                    self.showLabel.emit(
                        f'Integrating {len(pending)} frames (partial batch)...'
                    )
                _t_disp = time.time()
                files_processed += self._dispatch_batch(sphere, pending)
                if self.batch_mode or _t_read_accum >= 0.5:
                    logger.info(
                        '[FLUSH] %d frames  read=%.2fs  dispatch=%.2fs',
                        len(pending), _t_read_accum, time.time() - _t_disp,
                    )
                pending = []
                _t_read_accum = 0.0

        # Process whatever is left.  force_save=True so any remaining
        # _frames_since_save tail in live mode is flushed to disk.
        if pending and sphere is not None and self.command != 'stop':
            _t_disp = time.time()
            files_processed += self._dispatch_batch(
                sphere, pending, force_save=True,
            )
            logger.info(
                '[FLUSH-FINAL] %d frames  read=%.2fs  dispatch=%.2fs',
                len(pending), _t_read_accum, time.time() - _t_disp,
            )
        elif (sphere is not None and not self.xye_only
              and self._frames_since_save > 0 and self.command != 'stop'):
            # Pending was empty but the live-save batcher has unflushed
            # frames (last batch hit the divisor exactly).  Force a save
            # before leaving the collect loop.
            _get_h5pool().pause(sphere.data_file)
            try:
                with self.file_lock:
                    sphere._save_to_nexus()
            finally:
                _get_h5pool().resume(sphere.data_file)
            self._flush_xye_buffer(sphere)
            self._frames_since_save = 0

        # ── Phase 3: live watching ────────────────────────────────────────────
        if self.live_mode and self.command != 'stop' and sphere is not None:
            self.showLabel.emit('Watching for new files...')
            # Adaptive backoff between filesystem polls.  Starts tight
            # so first-frame latency is small (~100 ms vs the old fixed
            # 2 s — matters a lot for fast detectors like Eiger 4M);
            # doubles on each consecutive miss up to ``_POLL_MAX`` so
            # an idle wait doesn't burn CPU.  Resets on any hit so a
            # steady-state acquisition stays at the low end of the
            # range.  No external watcher dep — works on every FS the
            # beamline mounts (NFS, SMB, local).
            _poll_min = 0.1
            _poll_max = 2.0
            _poll_growth = 2.0
            poll_s = _poll_min
            while self.command != 'stop':
                img_file, scan_name, img_number, img_data, img_meta = self.get_next_image()
                if img_data is None:
                    # Nothing new yet — show watching status and sleep
                    self.showLabel.emit('Watching for new files...')
                    time.sleep(poll_s)
                    poll_s = min(poll_s * _poll_growth, _poll_max)
                    continue
                # Hit — reset the backoff so the next miss falls back
                # to the snappy 100 ms baseline.
                poll_s = _poll_min

                img_number = 1 if img_number is None else img_number
                self.scan_name = scan_name

                if (sphere is None) or (scan_name != sphere.name):
                    sphere = self.initialize_sphere()
                    _cached_poni = None

                if self.poni is not _cached_poni:
                    sphere._cached_integrator = poni_to_integrator(self.poni)
                    sphere._cached_fiber_integrator = None
                    _cached_poni = self.poni
                    self._cached_gi_incident_angle = None

                if img_number in sphere.arches.index:
                    continue

                bg_raw = self.get_background(img_file, img_number, img_meta)
                # Process immediately — single-threaded for low latency.
                # _process_one uses add_arch(batch_save=True), so the
                # disk save still rides on _frames_since_save below.
                self._process_one(sphere, img_file, img_number, img_data, img_meta, bg_raw)
                files_processed += 1
                self._frames_since_save += 1
                if self._frames_since_save >= self.LIVE_SAVE_INTERVAL and not self.xye_only:
                    _get_h5pool().pause(sphere.data_file)
                    try:
                        with self.file_lock:
                            sphere._save_to_nexus()
                    finally:
                        _get_h5pool().resume(sphere.data_file)
                    self._flush_xye_buffer(sphere)
                    self._frames_since_save = 0

        # Final flush on exit (live-watch tail or stop request) so the
        # last few frames aren't lost.
        if (sphere is not None and not self.xye_only
                and self._frames_since_save > 0):
            _get_h5pool().pause(sphere.data_file)
            try:
                with self.file_lock:
                    sphere._save_to_nexus()
            finally:
                _get_h5pool().resume(sphere.data_file)
            self._flush_xye_buffer(sphere)
            self._frames_since_save = 0

        # In batch mode, emit a single final signal so the GUI can refresh
        if self.batch_mode and files_processed > 0:
            self.sigUpdate.emit(-1)
        logger.info('Total Files Processed: %d', files_processed)

    # ── Batch dispatch ────────────────────────────────────────────────────────

    def _dispatch_batch(self, sphere, pending, *, force_save=False):
        """Process a list of pending images — parallel in batch mode, serial otherwise.

        ``force_save`` only affects the serial path (live mode); the
        parallel path always saves at end of batch by construction
        (the whole point of batch mode is one big save per dispatch).
        """
        if self.batch_mode and len(pending) > 1:
            return self._dispatch_batch_parallel(sphere, pending)
        return self._dispatch_batch_serial(sphere, pending, force_save=force_save)

    # ``_resolve_arch_mask``, ``_apply_threshold_inline``, and
    # ``_flush_xye_buffer`` moved to wranglerThread (the base class)
    # in May 2026 — both specThread and nexusThread inherit them now.
    # See xdart/gui/tabs/static_scan/wranglers/wrangler_widget.py.

    def _dispatch_batch_serial(self, sphere, pending, *, force_save=False):
        """Sequential dispatch (live mode or single-image batches).

        Each frame in ``pending`` gets integrated + GUI-updated
        immediately (via ``_process_one``).  The v2 file save runs
        *only* when at least :attr:`LIVE_SAVE_INTERVAL` frames have
        accumulated since the last save, or when ``force_save=True``
        (used by the final-flush path and by the live-watch tail).
        That keeps per-frame stalls off the integration loop while
        bounding scan-state loss on a crash to ~8 frames.
        """
        count = 0
        for item in pending:
            if self.command == 'stop':
                break
            # item is (img_file, img_number, img_data, img_meta, bg_raw, t_read)
            self._process_one(sphere, *item)
            count += 1

        self._frames_since_save += count

        should_save = (
            not self.xye_only
            and (force_save or self._frames_since_save >= self.LIVE_SAVE_INTERVAL)
        )
        if should_save and self._frames_since_save > 0:
            _t_save0 = time.time()
            _get_h5pool().pause(sphere.data_file)
            try:
                with self.file_lock:
                    sphere._save_to_nexus()
            finally:
                _get_h5pool().resume(sphere.data_file)
            _t_save = time.time() - _t_save0
            _t_xye0 = time.time()
            self._flush_xye_buffer(sphere)
            _t_xye = time.time() - _t_xye0
            logger.info(
                '[SAVE] %d frames since last save  save=%.3fs  xye=%.3fs',
                self._frames_since_save, _t_save, _t_xye,
            )
            self._frames_since_save = 0
        return count

    def _prewarm_fiber_integrator_spec(self, sphere, pending_entry) -> None:
        """Build ``sphere._cached_fiber_integrator`` from the first
        pending frame.

        O5: mirrors :meth:`nexusThread._prewarm_fiber_integrator` —
        builds a throw-away arch from the first frame, computes its
        incidence angle, and seeds the fiber integrator on the
        sphere.  After this call ``_dispatch_batch_parallel`` can
        build a fiber pool so every worker borrows a deep-copied
        instance instead of constructing a fresh one per frame.

        Pre-O5 the parallel batch path left ``_cached_fiber_integrator``
        unset, ``fiber_pool`` came back None, and each worker built
        its own fiber integrator per frame (~250 ms first-call CSR
        LUT cost per worker per frame).  Costly for GI scans with
        many cores.

        ``pending_entry`` is one tuple from ``pending``:
        ``(img_file, img_number, img_data, img_meta, bg_raw, t_read)``.
        """
        if not self.gi:
            return
        if sphere._cached_fiber_integrator is not None:
            return
        img_file, img_number, img_data, img_meta, bg_raw, _ = pending_entry
        # Build a scratch arch identical in shape to what
        # ``_process_one`` constructs so the angle math agrees.
        scratch = EwaldArch(
            img_number, img_data, poni=self.poni,
            scan_info=img_meta, static=True, gi=True,
            th_mtr=self.incidence_motor,
            sample_orientation=self.sample_orientation,
            tilt_angle=self.tilt_angle,
            series_average=self.series_average,
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
        # The per-frame angle-drift check in ``_integrate_one``
        # compares against this cached value; if a later frame's
        # incidence angle differs, that worker falls back to a
        # locally-built fiber integrator at the right angle.
        self._cached_gi_incident_angle = incident_angle
        # Q3: ``_borrow_fiber_integrator`` (the shared helper used
        # by parallel workers) looks the angle up on the *sphere*,
        # not on the wrangler instance, because it doesn't have a
        # handle to the wrangler.  The NeXus prewarm sets the
        # sphere attribute too; pre-Q3 the SPEC prewarm only set
        # the wrangler attribute, so the borrow helper always saw
        # cached_angle=None and built worker-local FiberIntegrators
        # per frame — defeating the pool we just constructed.
        sphere._cached_fiber_integrator_angle = incident_angle

    def _dispatch_batch_parallel(self, sphere, pending):
        """Parallel batch processing using ThreadPoolExecutor.

        Phase 1 — Parallel integration:
            Each worker creates an EwaldArch, runs integrate_1d / integrate_2d,
            and writes xye/csv.  pyFAI's Cython integration releases the GIL,
            so threads get true parallelism for the CPU-heavy part.

        Phase 2 — Serial HDF5 write:
            All completed arches are written to HDF5 under a single file_lock
            acquisition.  Skipped entirely in xye_only mode.
        """
        n_workers = min(self.max_cores, len(pending))
        # Per-scan pool of N integrator copies — required because
        # pyFAI's AzimuthalIntegrator isn't thread-safe across workers
        # with different inputs.  See xdart.utils.integrator_pool.
        integrator_pool = ensure_integrator_pool(
            sphere, '_cached_integrator', n_workers,
        )
        if integrator_pool is None:
            # No cached integrator yet — fall back to serial dispatch
            # (the source integrator gets built on the first call).
            return self._dispatch_batch_serial(sphere, pending)
        # F3: prewarm the bad-pixel mask on the main thread before
        # any worker reads it, so cache initialization isn't racy.
        # ``pending`` tuple shape: (img_file, img_number, img_data,
        # img_meta, bg_raw, t_read) — index 2 is the image.
        if (getattr(sphere, '_cached_data_mask', None) is None
                and pending):
            self._prewarm_arch_mask(sphere, pending[0][2])
        # O5: prewarm the fiber integrator from the first pending
        # frame so the per-worker fiber_pool below is non-None.
        # Without this every worker built its own fiber integrator
        # per frame (slow, especially for the first frame of each
        # worker where pyFAI's CSR LUT cost is paid).  Mirrors the
        # ``_prewarm_fiber_integrator`` path the NeXus wrangler
        # already runs before its parallel section.
        if (self.gi
                and sphere._cached_fiber_integrator is None
                and pending):
            try:
                self._prewarm_fiber_integrator_spec(sphere, pending[0])
            except Exception as e:
                # Fall through: workers still build their own fiber
                # integrators per frame, just without the speedup.
                logger.debug(
                    "SPEC GI fiber prewarm failed (%s); workers will "
                    "build their own per-frame", e,
                )
        # H2: per-worker fiber-integrator pool (matches IntegratorPool
        # for the regular AzimuthalIntegrator).  None when GI is off
        # or when the prewarm hasn't seeded _cached_fiber_integrator
        # yet — _borrow_fiber_integrator falls back to the per-frame
        # build in those cases.
        fiber_pool = (
            ensure_integrator_pool(
                sphere, '_cached_fiber_integrator', n_workers,
                pool_attr='_cached_fiber_integrator_pool',
            )
            if self.gi and sphere._cached_fiber_integrator is not None
            else None
        )
        # Capture the angle the prewarmed fiber integrator was built
        # for.  When ``gi`` is on and a per-frame arch's incidence
        # angle drifts from this (i.e. ω varies across the scan, as
        # opposed to a sin²ψ-style fixed-ω χ/φ scan), the worker has
        # to fall back to a worker-local fiber integrator built at
        # the correct angle.  Reusing the prewarm would silently
        # integrate every frame as if it were at frame 0's incidence.
        # 1e-4 deg is below the noise floor of beamline motor readouts
        # and well below pyFAI's solid-angle sensitivity.
        cached_gi_angle = self._cached_gi_incident_angle
        _GI_ANGLE_TOL = 1e-4
        skip_2d = sphere.skip_2d
        bai_1d_args = dict(sphere.bai_1d_args)
        bai_2d_args = dict(sphere.bai_2d_args)
        mask = self.mask
        gi = self.gi
        standard_plan = self._plan_cache.get(sphere, integrate_2d=not skip_2d)
        th_mtr = self.incidence_motor
        sample_orientation = self.sample_orientation
        tilt_angle = self.tilt_angle
        series_average = self.series_average

        def _integrate_one(img_file, img_number, img_data, img_meta, bg_raw,
                           t_read=0.0):
            """Pure integration + xye write — no shared mutable state.

            ``t_read`` is accepted for tuple-shape compatibility with
            _process_one but isn't surfaced — parallel batch mode logs
            its read times via the per-batch [FLUSH] line.

            F2 cancel-fast: returns ``None`` immediately when Stop
            has been requested, so already-running workers exit
            before they hit pyFAI (the expensive part).
            """
            if self.command == 'stop':
                return None
            _t0 = time.time()
            # Threshold filtering: replace out-of-band pixels with the
            # dummy sentinel in a fresh float32 copy.  The arch mask
            # stays stable (cached from frame 1) so pyFAI's CSR engine
            # cache survives across frames.
            img_data = self._apply_threshold_inline(img_data)
            arch_mask = self._resolve_arch_mask(sphere, img_data)

            # Borrow a private integrator for the duration of this
            # frame's integration.  When the worker exits the `with`
            # block the integrator returns to the pool for the next
            # frame to grab.  Each integrator is touched by at most
            # one worker at a time — that's what makes parallel
            # batch correct vs the old shared-instance code path.
            with integrator_pool.borrow() as ai:
                arch = EwaldArch(
                    img_number, img_data, poni=self.poni,
                    scan_info=img_meta, static=True, gi=gi,
                    th_mtr=th_mtr, bg_raw=bg_raw,
                    sample_orientation=sample_orientation,
                    tilt_angle=tilt_angle,
                    series_average=series_average,
                    integrator=ai,
                    mask=arch_mask,
                )

                # H2 + S3 unified dispatch.  When ``standard_plan`` is
                # None (GI sphere) we run the fiber-integrator path
                # inside the legacy_gi closure; otherwise the headless
                # ``reduce_ewald_arch`` handles it.
                def _legacy_gi_for_arch() -> None:
                    with self._borrow_fiber_integrator(
                        sphere, fiber_pool, arch,
                    ) as _fi:
                        arch.integrate_1d(
                            global_mask=mask,
                            fiber_integrator=_fi,
                            **bai_1d_args,
                        )
                        if not skip_2d:
                            arch.integrate_2d(
                                global_mask=mask,
                                fiber_integrator=_fi,
                                **bai_2d_args,
                            )

                dispatch_arch_reduction(
                    arch, sphere,
                    standard_plan=standard_plan,
                    integrator=ai,
                    global_mask=mask,
                    legacy_gi=_legacy_gi_for_arch,
                )

            # Detach the pool integrator from this arch — once the
            # `with` block exited, the next worker can borrow this
            # same instance and start mutating it.  Replace with the
            # sphere's source integrator (which the pool never hands
            # out, so it's safe to share with non-parallel consumers).
            arch.integrator = sphere._cached_integrator

            # Precompute the raw-image thumbnail here, in parallel with
            # other workers' integrations, rather than on the serial Phase 2
            # writer thread.  scipy.ndimage.zoom and numpy subtract release
            # the GIL for their C code, so this overlaps cleanly.
            try:
                arch.make_thumbnail(global_mask=mask)
            except Exception as e:
                logger.warning('Thumbnail precompute failed for image %s: %s',
                               img_number, e)

            # Buffer the XYE write — flushed at end of batch by the
            # serial dispatcher.  Keeps the worker thread cheap and
            # groups disk traffic so it doesn't interleave with the
            # next batch's integration.
            with self._xye_lock:
                self._xye_buffer.append((img_number, arch))

            _elapsed = time.time() - _t0
            fname = os.path.splitext(os.path.basename(img_file))[0]
            logger.info('[PARALLEL] image_%04d (%s): %.2fs', img_number, fname[-30:], _elapsed)
            return arch

        # ── Phase 1: parallel integration ────────────────────────────────────
        # Shared base helper handles ThreadPoolExecutor wiring,
        # stop-flag honoring, per-worker exception logging, and
        # idx-sort.  Both SPEC batch and NeXus chunked dispatch now
        # share this primitive.
        self.showLabel.emit(f'Integrating {len(pending)} images ({n_workers} workers)...')
        _t_phase1 = time.time()
        arches = self._parallel_integrate(
            pending,
            lambda item: _integrate_one(*item),
            n_workers,
            label='BATCH',
        )
        _t_phase1 = time.time() - _t_phase1
        logger.info('[BATCH] Phase 1 (parallel integration): %d frames in %.2fs',
                    len(arches), _t_phase1)

        if not arches:
            return 0

        # ── Phase 2: serial HDF5 batch write ─────────────────────────────────
        if not self.xye_only:
            self.showLabel.emit(f'Writing {len(arches)} frames to HDF5...')
            _t_phase2 = time.time()
            # img_number → img_file lookup for NeXus provenance.  Keyed
            # on arch.idx (== img_number) so the lookup matches the
            # arch-only result list returned by _parallel_integrate.
            _img_files = {item[1]: item[0] for item in pending}
            _get_h5pool().pause(sphere.data_file)
            try:
                with self.file_lock:
                    # Phase 2a: in-memory accumulation only (batch_save=True
                    # makes add_arch a pure in-memory op; no file I/O).
                    for arch in arches:
                        img_number = arch.idx
                        img_file = _img_files.get(img_number, '')
                        if img_file:
                            try:
                                arch.source_file = os.path.relpath(
                                    img_file, os.path.dirname(sphere.data_file))
                            except ValueError:
                                arch.source_file = str(img_file)
                        else:
                            arch.source_file = ""
                        # source_frame_idx is the *per-source-file*
                        # 0-based frame offset, which lazy raw load
                        # passes to NexusImageStack(source_file)[idx].
                        # • Single-frame sources (TIF/EDF/CBF/…) →
                        #   source IS the frame; idx is always 0.
                        # • Multi-frame sources (Eiger / NeXus master)
                        #   → use img_number-1, since
                        #   ``_get_next_eiger_frame_sync`` emits
                        #   ``img_number = frame_idx + 1`` (1-based).
                        # Pre-C1 this was hardcoded to 0, so any
                        # reload + lazy raw read on Eiger always
                        # returned frame 0 — silently wrong.
                        if _raw_lives_in_source(img_file):
                            arch.source_frame_idx = int(img_number) - 1
                        else:
                            arch.source_frame_idx = 0
                        arch.skip_map_raw = skip_2d or _raw_lives_in_source(img_file)

                        sphere.add_arch(
                            arch=arch, calculate=False, update=True,
                            get_sd=True, set_mg=False, static=True, gi=gi,
                            th_mtr=th_mtr, series_average=series_average,
                            batch_save=True,
                        )
                    # Phase 2b: single batch flush — one slice-assign per
                    # stacked dataset for all frames in this batch.
                    # The writer owns its file handle now.
                    sphere._save_to_nexus()
            finally:
                _get_h5pool().resume(sphere.data_file)
            _t_phase2 = time.time() - _t_phase2
            logger.info('[BATCH] Phase 2 (HDF5 write): %d frames in %.2fs', len(arches), _t_phase2)

        # Flush buffered XYE writes for this batch.  P3: pass the
        # set of arch.idx values that actually landed in .nxs so
        # in-flight workers from a Stop'd batch don't leave orphan
        # XYE files for frames that were never published.
        _t_xye = time.time()
        published_idxs = {a.idx for a in arches}
        self._flush_xye_buffer(sphere, published_idxs=published_idxs)
        _t_xye = time.time() - _t_xye
        if _t_xye > 0.01:
            logger.info('[BATCH] XYE flush: %d frames in %.2fs', len(arches), _t_xye)

        return len(arches)

    def _process_one(self, sphere, img_file, img_number, img_data, img_meta,
                     bg_raw, t_read=0.0):
        """Integrate one image sequentially and save. Includes timing instrumentation.

        ``t_read`` is the per-frame wall-clock time spent reading the
        image off disk (or off the prefetch queue) — measured by the
        collect loop and threaded through so the [TIMING] line shows
        per-frame variance.  Defaults to 0 for callers that don't
        track it (the live-watch loop reads via a different path).
        """
        fname = os.path.splitext(os.path.basename(img_file))[0]
        # Multi-frame containers reuse a single master filename across
        # frames; appending "[frame N]" to the status box so the user
        # can see the frame index advancing during the scan.
        _ext = Path(img_file).suffix.lower()
        if _ext in ('.h5', '.hdf5', '.nxs') or _is_eiger_master(img_file):
            self.showLabel.emit(
                f'Processing {self._middle_truncate(fname)} [frame {img_number}]'
            )
        else:
            self.showLabel.emit(f'Processing {self._middle_truncate(fname)}')

        _t1 = time.time()
        # Threshold via dummy sentinel + stable cached mask — see
        # _apply_threshold_inline / _resolve_arch_mask docstrings for
        # why this is fast even with per-frame threshold filtering on.
        img_data = self._apply_threshold_inline(img_data)
        arch_mask = self._resolve_arch_mask(sphere, img_data)
        arch = EwaldArch(
            img_number, img_data, poni=self.poni,
            scan_info=img_meta, static=True, gi=self.gi,
            th_mtr=self.incidence_motor, bg_raw=bg_raw,
            sample_orientation=self.sample_orientation,
            tilt_angle=self.tilt_angle,
            series_average=self.series_average,
            integrator=sphere._cached_integrator,
            mask=arch_mask,
        )
        _t_arch = time.time() - _t1

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

        _t2 = time.time()

        def _legacy_gi_for_single() -> None:
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

        dispatch_arch_reduction(
            arch, sphere,
            standard_plan=self._plan_cache.get(
                sphere, integrate_2d=not sphere.skip_2d,
            ),
            integrator=sphere._cached_integrator,
            global_mask=self.mask,
            legacy_gi=_legacy_gi_for_single,
        )
        # Timing kept for parity with the legacy logging; the
        # standard path now does both 1D + 2D in one call so we
        # bundle the total under _t_1d.
        _t_1d = time.time() - _t2
        _t_2d = 0.0

        # ── GUI data (skip in batch mode — no one is looking) ────────────
        _t_h5_total = _t_h5_wait = _t_h5_write = 0.0
        if not self.batch_mode:
            self.data_1d[int(img_number)] = arch.copy(include_2d=False)
            self.data_2d[int(img_number)] = {
                'map_raw': arch.map_raw,
                'bg_raw': arch.bg_raw,
                'mask': arch.mask,
                'int_2d': arch.int_2d,
                'gi_2d': arch.gi_2d,
                'thumbnail': None,
            }

        # ── In-memory accumulation (no disk I/O — batch flush handles that) ──
        if not self.xye_only:
            # Set source file as relative path from HDF5 dir for NeXus provenance
            if img_file:
                try:
                    arch.source_file = os.path.relpath(
                        img_file, os.path.dirname(sphere.data_file))
                except ValueError:
                    arch.source_file = str(img_file)
            else:
                arch.source_file = ""
            # source_frame_idx: per-source-file 0-based frame offset.
            # See the matching block in ``_dispatch_batch_parallel``
            # for the full rationale.  Eiger / HDF5 masters get
            # ``img_number - 1``; everything else stays at 0.
            if _raw_lives_in_source(img_file):
                arch.source_frame_idx = int(img_number) - 1
            else:
                arch.source_frame_idx = 0
            # For Eiger: raw frames already live in the master file — don't double-store them.
            arch.skip_map_raw = sphere.skip_2d or _raw_lives_in_source(img_file)
            _t4 = time.time()
            # batch_save=True → pure in-memory (stash + index + bai_*).
            # The serial dispatcher calls sphere._save_to_nexus once at
            # end-of-batch, so we don't pay per-frame write cost here.
            sphere.add_arch(
                arch=arch, calculate=False, update=True,
                get_sd=True, set_mg=False, static=True, gi=self.gi,
                th_mtr=self.incidence_motor, series_average=self.series_average,
                batch_save=True,
            )
            _t_h5_total = time.time() - _t4
            _t_h5_wait = 0.0
            _t_h5_write = _t_h5_total

        # ── XYE buffer (flushed at end of batch by the dispatcher) ──────
        _t5 = time.time()
        with self._xye_lock:
            self._xye_buffer.append((img_number, arch))
        _t_csv = time.time() - _t5

        _t_total = t_read + _t_arch + _t_1d + _t_2d + _t_h5_total + _t_csv
        # Merged per-frame line: timing + the user-facing "processed
        # <file>" annotation.  3-decimal precision so sub-10ms
        # components (in-memory add_arch, XYE buffer append) don't
        # round to 0.00.  For multi-frame containers (.h5/.hdf5/.nxs)
        # the label is "<master> frame N" — the frame number is
        # already explicit there, so we drop the redundant
        # "image_NNNN" prefix that we keep for per-file inputs.
        _ext = Path(img_file).suffix.lower()
        if _ext in ('.h5', '.hdf5', '.nxs'):
            _label = f'{fname} frame {img_number}'
        else:
            _label = f'image_{img_number:04d} {fname}'
        _sub = f' {self.sub_label}' if self.sub_label else ''
        logger.info(
            '[TIMING] %s: read=%.3fs arch_init=%.3fs '
            'int_1d=%.3fs int_2d=%.3fs add_arch=%.3fs csv=%.3fs '
            'total=%.3fs%s',
            _label, t_read, _t_arch, _t_1d, _t_2d,
            _t_h5_total, _t_csv, _t_total, _sub,
        )
        # In batch mode, suppress per-frame GUI signals — emit once at end
        if not self.batch_mode:
            # Publish the freshly-integrated arch so the main thread can
            # consume it without going back to disk.  See
            # static_scan_widget.update_data for the consumer side.
            self._published_arches[img_number] = arch
            self.sigUpdate.emit(img_number)

    # ── Eiger HDF5 helpers ────────────────────────────────────────────────

    def _get_nframes(self, master_path):
        """Return frame count for a master file, 0 on failure."""
        return count_frames(master_path)

    def _eiger_open_master(self, master_path):
        """Open (or switch to) an Eiger / NeXus HDF5 file, keeping the handle.

        Strategy
        --------
        - ``.nxs`` files → skip fabio and go straight to h5py using
          ``find_nexus_image_dataset`` to locate the 3D image dataset.
          fabio's EigerImage is tuned for Eiger master layouts and does
          not reliably find image arrays in Bluesky-style NeXus files.
        - Otherwise (``_master.h5`` etc.):
            1. **Persistent fabio handle (primary)** — ``fabio.EigerImage``
               is purpose-built for Eiger master files and handles all
               firmware variants, external-link layouts (``_data_*.h5``),
               and frame indexing natively.
            2. **h5py fallback** — if fabio fails, locate the 3D image
               dataset via ``find_nexus_image_dataset``.
        """
        self._eiger_close_master()

        ext = Path(master_path).suffix.lower()
        is_nexus = ext == '.nxs'

        if not is_nexus:
            try:
                # Primary: fabio (handles all Eiger layouts)
                self._eiger_fabio_handle = fabio.open(master_path)
                self._eiger_nframes = self._eiger_fabio_handle.nframes
                return
            except (IOError, OSError) as e:
                logger.debug("Failed to open %s with fabio: %s, trying h5py", master_path, e)
                self._eiger_fabio_handle = None

        # h5py path (primary for .nxs, fallback for .h5/.hdf5 master files)
        try:
            ds_path = find_nexus_image_dataset(master_path)
            if ds_path is None:
                logger.warning('Could not find 3D image dataset in %s', master_path)
                self._eiger_close_master()
                self._eiger_nframes = 0
                return
            self._eiger_h5_handle = h5py.File(master_path, 'r')
            self._eiger_h5_dataset = self._eiger_h5_handle[ds_path]
            self._eiger_nframes = self._eiger_h5_dataset.shape[0]
        except Exception as e:
            logger.error('Error opening HDF5/NeXus file %s: %s', master_path, e)
            self._eiger_close_master()
            self._eiger_nframes = 0

    def _eiger_close_master(self):
        """Close persistent Eiger handles (fabio and/or h5py)."""
        self._eiger_h5_dataset = None
        if self._eiger_fabio_handle is not None:
            try:
                self._eiger_fabio_handle.close()
            except (IOError, OSError) as e:
                logger.debug("Failed to close fabio handle: %s", e)
            self._eiger_fabio_handle = None
        if self._eiger_h5_handle is not None:
            try:
                self._eiger_h5_handle.close()
            except (IOError, OSError) as e:
                logger.debug("Failed to close h5py handle: %s", e)
            self._eiger_h5_handle = None

    def _eiger_refill_master_queue(self):
        """Glob for *_master.h5 files not yet processed (Image Directory mode)."""
        filters = '*' + '*'.join(f for f in self.file_filter.split()) + '*'
        filters = filters if filters != '**' else '*'
        pattern = f'{filters}_master.h5'
        if self.include_subdir:
            master_files = sorted(Path(self.img_dir).rglob(pattern))
        else:
            master_files = sorted(Path(self.img_dir).glob(pattern))
        queued = set(self._eiger_master_queue)
        for mf in master_files:
            mf_str = str(mf)
            if mf_str not in self._eiger_done_masters and mf_str not in queued:
                self._eiger_master_queue.append(mf_str)

    def _get_next_eiger_frame(self):
        """Return the next frame from Eiger / NeXus HDF5 file(s).

        Wraps :meth:`_get_next_eiger_frame_sync` with a background
        prefetcher so the next frame's disk read overlaps with the
        current frame's integration.  The synchronous reader is still
        available for the worker itself and for paths that don't want
        prefetching.

        Uses a short polling timeout so the main thread can break out
        of a blocked ``.get()`` if the user hits Stop and the prefetch
        worker exited without pushing a sentinel.
        """
        if self._prefetch_queue is None:
            self._start_prefetcher()
        # Poll the queue so we can cooperate with user Stop even if the
        # prefetcher died or hasn't pushed an end-of-stream sentinel yet.
        while True:
            if self.command == 'stop':
                return (None, None, 1, None, {})
            try:
                item = self._prefetch_queue.get(timeout=0.25)
            except queue.Empty:
                # If the worker is gone and queue is drained, fall through
                # with an end-of-stream sentinel so the caller can exit.
                if (self._prefetch_thread is None
                        or not self._prefetch_thread.is_alive()):
                    return (None, None, 1, None, {})
                continue
            # O8: surface a worker-failure sentinel as a user-visible
            # status before propagating end-of-stream.  A clean end
            # has ``_prefetch_error is None``; a worker crash has the
            # error string set in the worker's except branch.
            if item[3] is None and self._prefetch_error:
                self.showLabel.emit(
                    f'Eiger read failed: {self._prefetch_error}'
                )
                # Clear so the next start (if any) doesn't re-emit.
                self._prefetch_error = None
            return item

    def _start_prefetcher(self):
        """Spin up the background prefetch thread (idempotent)."""
        if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
            return
        self._prefetch_queue = queue.Queue(maxsize=_PREFETCH_QUEUE_SIZE)
        self._prefetch_stop_evt = threading.Event()
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            name='eiger-prefetch',
            daemon=True,
        )
        self._prefetch_thread.start()

    def _push_frame_to_queue(self, item):
        """Put *item* onto the prefetch queue, cooperating with stop.

        Returns True if the item was queued, False if the worker was
        cancelled while blocking on a full queue.
        """
        while not self._prefetch_stop_evt.is_set() and self.command != 'stop':
            try:
                self._prefetch_queue.put(item, timeout=0.25)
                return True
            except queue.Full:
                continue
        return False

    def _prefetch_worker(self):
        """Read frames sequentially and push them onto the bounded queue.

        Uses bulk HDF5 slices (`dset[i:i+N]`) where possible so each chunk
        is decompressed once per N frames instead of N times.  Falls back
        to single-frame reads via :meth:`_get_next_eiger_frame_sync` for
        the initial master-file setup, fabio-backed sources, and
        live-growing files (where bulk reads past the known end are
        unsafe).

        The worker exits when:
          - a frame with ``img_data is None`` is produced (end of stream);
          - the stop event is set (cooperative cancellation);
          - ``self.command == 'stop'`` (user pressed Stop).
        Exceptions are logged and a sentinel tuple is pushed so the
        consumer can terminate cleanly.
        """
        sentinel_pushed = False
        try:
            while not self._prefetch_stop_evt.is_set() and self.command != 'stop':
                # Always fetch the first frame of (the next) master through
                # the sync reader — it handles master-queue advancement,
                # handle opening, and frame-count refresh.
                item = self._get_next_eiger_frame_sync()
                if not self._push_frame_to_queue(item):
                    return
                if item[3] is None:
                    # End of stream; worker is done — sentinel already queued.
                    sentinel_pushed = True
                    return

                # Fast path: bulk-read the remainder of this master through
                # the h5py dataset.  Skipped when fabio is primary (fabio
                # handles its own per-frame decoding) or when we're near
                # the tail of a live-growing file.
                while (not self._prefetch_stop_evt.is_set()
                       and self.command != 'stop'
                       and self._eiger_fabio_handle is None
                       and self._eiger_h5_dataset is not None
                       and self._eiger_frame_idx < self._eiger_nframes):

                    start = self._eiger_frame_idx
                    end = min(start + _PREFETCH_READ_CHUNK, self._eiger_nframes)
                    _t_blk = time.time()
                    try:
                        block = np.asarray(
                            self._eiger_h5_dataset[start:end], dtype='int32'
                        )
                    except Exception as e:
                        logger.warning(
                            'Bulk read failed (start=%d end=%d): %s; '
                            'falling back to per-frame read',
                            start, end, e,
                        )
                        break  # outer loop will resume with sync reader
                    _t_blk = time.time() - _t_blk
                    # If a bulk read takes >50ms it can fight with the
                    # consumer for memory bandwidth — log so we can
                    # correlate against [TIMING] spikes on the consumer.
                    if _t_blk > 0.05:
                        logger.info(
                            '[PREFETCH] block frames %d-%d read in %.3fs',
                            start, end - 1, _t_blk,
                        )

                    meta = (read_image_metadata(self._eiger_master_path,
                                                meta_format=self.meta_ext)
                            if self.meta_ext else {})
                    master_stem = Path(self._eiger_master_path).stem
                    scan_name = (master_stem[:-7]
                                 if master_stem.endswith('_master')
                                 else master_stem)

                    # Advance the shared frame cursor *before* dispatching so
                    # that any concurrent sync read (e.g. in fallback) does
                    # not re-serve these frames.
                    self._eiger_frame_idx = end

                    for i in range(end - start):
                        if (self._prefetch_stop_evt.is_set()
                                or self.command == 'stop'):
                            return
                        frame_idx = start + i
                        item = (
                            self._eiger_master_path,
                            scan_name,
                            frame_idx + 1,   # 1-based img_number
                            block[i],
                            meta,
                        )
                        if not self._push_frame_to_queue(item):
                            return
        except Exception as e:
            logger.exception('Eiger prefetch worker failed: %s', e)
            # O8: stamp the failure so the consumer of the sentinel
            # can tell "scan ended cleanly" from "worker crashed".
            # Surfaced via showLabel in _get_next_eiger_frame; without
            # this the user sees a silent end-of-scan and assumes
            # acquisition completed normally.
            self._prefetch_error = str(e)
        finally:
            # Guarantee the consumer unblocks no matter how we exit
            # (stop event, command=='stop', exception, or normal return).
            if not sentinel_pushed:
                try:
                    self._prefetch_queue.put(
                        (None, None, 1, None, {}), timeout=1.0,
                    )
                except queue.Full:
                    # Drain one slot so the sentinel can fit — the main
                    # thread has already seen stop, so dropping a queued
                    # frame is fine.
                    try:
                        self._prefetch_queue.get_nowait()
                        self._prefetch_queue.put_nowait(
                            (None, None, 1, None, {}),
                        )
                    except (queue.Empty, queue.Full):
                        pass

    def _prefetch_stop_prior(self):
        """Cancel any prior prefetch thread and drain its queue."""
        if self._prefetch_stop_evt is not None:
            self._prefetch_stop_evt.set()
        if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
            # Drain to unblock the worker on a full queue
            if self._prefetch_queue is not None:
                try:
                    while True:
                        self._prefetch_queue.get_nowait()
                except queue.Empty:
                    pass
            self._prefetch_thread.join(timeout=2.0)

    def _get_next_eiger_frame_sync(self):
        """Return the next frame from Eiger HDF5 master file(s), one at a time.

        Keeps the h5py file handle open across frames to avoid the
        expensive open/close cycle per frame.  Tracks position with
        (_eiger_master_path, _eiger_frame_idx, _eiger_nframes).
        """
        # ── Initialise on the very first call ────────────────────────────────
        if self._eiger_master_path is None:
            if self.inp_type == 'Image Directory':
                self._eiger_refill_master_queue()
                if not self._eiger_master_queue:
                    return None, None, 1, None, {}
                self._eiger_master_path = self._eiger_master_queue.popleft()
            else:
                self._eiger_master_path = self.img_file
            self._eiger_frame_idx = 0
            self._eiger_open_master(self._eiger_master_path)
            if self._eiger_nframes == 0:
                self._eiger_close_master()
                return None, None, 1, None, {}

        # ── Current master exhausted?  Try to advance ────────────────────────
        if self._eiger_frame_idx >= self._eiger_nframes:
            # Re-check frame count (file may still be growing in live mode)
            if self._eiger_fabio_handle is not None:
                # Reopen fabio handle to pick up newly written data files
                try:
                    self._eiger_fabio_handle.close()
                    self._eiger_fabio_handle = fabio.open(self._eiger_master_path)
                    self._eiger_nframes = self._eiger_fabio_handle.nframes
                except (IOError, OSError) as e:
                    logger.debug("Failed to reopen fabio handle for %s: %s", self._eiger_master_path, e)
                    self._eiger_nframes = self._get_nframes(self._eiger_master_path)
            elif self._eiger_h5_dataset is not None:
                try:
                    self._eiger_h5_dataset.id.refresh()
                    self._eiger_nframes = self._eiger_h5_dataset.shape[0]
                except (IOError, OSError, KeyError) as e:
                    logger.debug("Failed to refresh h5 dataset for %s: %s", self._eiger_master_path, e)
                    self._eiger_nframes = self._get_nframes(self._eiger_master_path)
            else:
                self._eiger_nframes = self._get_nframes(self._eiger_master_path)

        if self._eiger_frame_idx >= self._eiger_nframes:
            if self.inp_type == 'Image Directory':
                self._eiger_done_masters.add(self._eiger_master_path)
                self._eiger_close_master()
                if not self._eiger_master_queue:
                    self._eiger_refill_master_queue()
                if not self._eiger_master_queue:
                    return None, None, 1, None, {}
                self._eiger_master_path = self._eiger_master_queue.popleft()
                self._eiger_frame_idx = 0
                self._eiger_open_master(self._eiger_master_path)
                if self._eiger_nframes == 0:
                    self._eiger_close_master()
                    return None, None, 1, None, {}
            else:
                self._eiger_close_master()
                return None, None, 1, None, {}

        # ── Read one frame ────────────────────────────────────────────────────
        frame_idx = self._eiger_frame_idx
        self._eiger_frame_idx += 1

        try:
            if self._eiger_fabio_handle is not None:
                # Primary: persistent fabio handle
                _raw = (self._eiger_fabio_handle.data if frame_idx == 0
                        else self._eiger_fabio_handle.get_frame(frame_idx).data)
                img_data = np.asarray(_raw, dtype='int32')
            elif self._eiger_h5_dataset is not None:
                # Fallback: h5py dataset
                img_data = np.asarray(self._eiger_h5_dataset[frame_idx], dtype='int32')
            else:
                # Should not happen, but safety net
                with fabio.open(self._eiger_master_path) as _img:
                    _raw = _img.data if frame_idx == 0 else _img.get_frame(frame_idx).data
                img_data = np.asarray(_raw, dtype='int32')
        except Exception as e:
            logger.error('Error reading frame %d from %s: %s', frame_idx, self._eiger_master_path, e)
            img_data = None

        meta = read_image_metadata(self._eiger_master_path, meta_format=self.meta_ext) if self.meta_ext else {}

        master_stem = Path(self._eiger_master_path).stem
        scan_name = master_stem[:-7] if master_stem.endswith('_master') else master_stem
        img_number = frame_idx + 1  # 1-based

        return self._eiger_master_path, scan_name, img_number, img_data, meta

    # ── Image iteration ──────────────────────────────────────────────────

    def get_next_image(self):
        """Gets next image in image series or in directory to process."""
        is_master = _is_eiger_master(self.img_file) if self.img_file else False

        if self.single_img and not is_master:
            img_data = np.asarray(read_image(self.img_file), dtype=float)
            meta = read_image_metadata(self.img_file, meta_format=self.meta_ext) if self.meta_ext else {}
            scan_name, img_number = _get_scan_info(self.img_file)
            return self.img_file, scan_name, img_number, img_data, meta

        if is_master or self.img_ext.lower() in ('h5', 'hdf5', 'nxs'):
            return self._get_next_eiger_frame()

        if len(self.img_fnames) == 0:
            if self.inp_type != 'Image Directory':
                first_img = self.img_file
                # Glob is loose: `{scan_name}_*.{ext}` would also match
                # neighbours like `{scan_name}_again_0001.{ext}`. Filter to
                # files whose tail is purely a numeric frame index, so we
                # only pick up the *strict* siblings of self.img_file.
                _series_re = re.compile(
                    rf'^{re.escape(self.scan_name)}_\d+\.{re.escape(self.img_ext)}$'
                )
                self.img_fnames = [
                    p for p in Path(self.img_dir).glob(
                        f'{self.scan_name}_*.{self.img_ext}'
                    )
                    if _series_re.match(p.name)
                ]
            else:
                first_img = ''
                filters = '*' + '*'.join(f for f in self.file_filter.split()) + '*'
                filters = filters if filters != '**' else '*'
                if self.include_subdir:
                    self.img_fnames = Path(self.img_dir).rglob(f'{filters}.{self.img_ext}')
                else:
                    self.img_fnames = Path(self.img_dir).glob(f'{filters}.{self.img_ext}')

            self.img_fnames = [str(f) for f in self.img_fnames if
                               (str(f) >= first_img) and (str(f) not in self.processed)]

            self.img_fnames = deque(natural_sort_ints(self.img_fnames))

        img_file, scan_name, img_number, img_data, img_meta = None, None, 1, None, {}
        n = 0
        while len(self.img_fnames) > 0:
            fname = self.img_fnames[0]
            sname, snumber = _get_scan_info(fname)

            if (n > 0) and (scan_name != sname):
                break

            self.processed.append(fname)
            self.img_fnames.popleft()

            data = np.asarray(read_image(fname), dtype=float)
            if data is None or not np.isfinite(data).any():
                continue

            meta = read_image_metadata(fname, meta_format=self.meta_ext) if self.meta_ext else {}
            n += 1

            if (not self.series_average) or (snumber is None):
                return fname, sname, snumber, data, meta
            else:
                if n == 1:
                    img_data = data
                    img_meta = meta
                else:
                    img_data += data
                    for (k, v) in meta.items():
                        try:
                            img_meta[k] = float(img_meta[k]) + float(meta[k])
                        except TypeError:
                            pass

                scan_name, img_file = sname, fname

        if n > 1:
            img_data /= n
            for (k, v) in img_meta.items():
                try:
                    img_meta[k] /= n
                except TypeError:
                    pass

        return img_file, scan_name, img_number, img_data, img_meta

    # ── Metadata / Background ────────────────────────────────────────────

    def get_meta_data(self, img_file):
        return read_image_metadata(img_file, meta_format=self.meta_ext)

    def subtract_bg(self, img_data, img_file, img_number, img_meta):
        bg = self.get_background(img_file, img_number, img_meta)
        try:
            img_data -= bg
        except ValueError:
            pass

    def initialize_sphere(self):
        """If scan changes, initialize new EwaldSphere object.
        If mode is overwrite, replace existing HDF5 file, else append to it.
        """
        fname = os.path.join(self.h5_dir, self.scan_name + '.nxs')
        # Eiger master files are pre-processed with the trailing
        # ``_master`` suffix stripped from scan_name (see
        # _get_next_eiger_frame). Without this sync, the wrangler
        # widget's self.fname (set from the original master filename
        # in spec_wrangler.setup()) diverges from the actual sphere
        # output path, and static_scan_widget.wrangler_finished
        # cannot find the generated file to reload at end of batch.
        self.fname = fname
        sphere = EwaldSphere(self.scan_name,
                             data_file=fname,
                             static=True,
                             gi=self.gi,
                             incidence_motor=self.incidence_motor,
                             series_average=self.series_average,
                             single_img=self.single_img,
                             global_mask=self.mask,
                             # J2: share lock with wrangler save path
                             file_lock=self.file_lock,
                             **self.sphere_args)
        sphere.skip_2d = self.sphere.skip_2d
        # v2 NeXus writer needs a DiffractometerGeometry to derive per-frame
        # rot1/rot2/rot3 and incidence-angle arrays from scan_data.  The
        # default is a two-circle convention using `tth` (detector arm) and
        # whatever `th_mtr` resolves to (sample tilt).  Override later from
        # the geometry UI panel when the user picks a non-default convention.
        sphere.default_geometry()

        write_mode = self.write_mode
        if not os.path.exists(fname):
            write_mode = 'Overwrite'

        _get_h5pool().pause(sphere.data_file)
        try:
            with self.file_lock:
                if write_mode == 'Append':
                    # v2 NeXus loader (the only one we support now).
                    sphere.load_from_h5(replace=False, mode='a')
                    sphere.skip_2d = self.sphere.skip_2d
                    for (k, v) in self.sphere_args.items():
                        setattr(sphere, k, v)
                    existing_arches = sphere.arches.index
                    if len(existing_arches) == 0:
                        sphere.save_to_nexus(replace=True)
                else:
                    sphere.save_to_nexus(replace=True)
        finally:
            _get_h5pool().resume(sphere.data_file)

        # Copy integration args (including GI modes) from the main sphere.
        sphere.bai_1d_args = self.sphere.bai_1d_args.copy()
        sphere.bai_2d_args = self.sphere.bai_2d_args.copy()

        self.sigUpdateFile.emit(
            self.scan_name, fname,
            self.gi, self.incidence_motor, self.single_img,
            self.series_average
        )
        logger.info('***** New Scan *****')
        logger.info('Output file: %s', fname)

        return sphere

    def get_mask(self):
        """Get mask array from mask file."""
        self.mask = self.detector.calc_mask()
        if self.mask_file and os.path.exists(self.mask_file):
            if self.mask is not None:
                try:
                    self.mask += fabio.open(self.mask_file).data
                except ValueError:
                    logger.warning('Mask file not valid for Detector (shape mismatch)')
                    pass
            else:
                self.mask = fabio.open(self.mask_file).data

        if self.mask is None:
            return None

        if self.mask.shape != self.detector.shape:
            logger.warning('Mask file not valid for Detector (shape %s != %s)', self.mask.shape, self.detector.shape)
            return None

        self.mask = np.flatnonzero(self.mask)

    def threshold(self, img_data):
        """Return flat indices of pixels outside [threshold_min, threshold_max]."""
        mask = (img_data < self.threshold_min) | (img_data > self.threshold_max)
        return np.flatnonzero(mask)

    def get_background(self, img_file, img_number, img_meta):
        """Subtract background image if bg_file or bg_dir specified."""
        if self.bg_type == 'None':
            return 0

        bg, bg_file, bg_meta, norm_factor = 0, None, None, 1
        self.sub_label, norm_label, bg_scale_label = '', '', ''

        if self.bg_type == 'Single BG File':
            if self.bg_file:
                bg_file = self.bg_file
                bg_meta = read_image_metadata(bg_file, meta_format=self.meta_ext)
        elif self.bg_type == 'Series Average':
            if self.bg_file:
                sname, fnames, bg, bg_meta = get_series_avg(self.bg_file, self.detector, self.meta_ext)
                if sname is None:
                    return 0
        else:
            if self.bg_dir and (self.bg_match_fname or self.bg_matching_par):
                bg_file_filter = 'bg' if not self.bg_file_filter else self.bg_file_filter
                if self.bg_match_fname:
                    bg_file_filter = f'{self.scan_name} {bg_file_filter}'
                filters = '*' + '*'.join(f for f in bg_file_filter.split()) + '*'
                filters = filters if filters != '**' else '*'

                meta_files = sorted(glob.glob(os.path.join(
                    self.img_dir, f'{filters}.{self.meta_ext}')))

                for meta_file in meta_files:
                    bg_file = f'{os.path.splitext(meta_file)[0]}.{self.img_ext}'
                    if bg_file == img_file:
                        bg_file = None
                        continue

                    bg_meta = read_image_metadata(bg_file, meta_format=self.meta_ext)
                    if self.bg_match_fname:
                        _, meta_img_num = _get_scan_info(meta_file)
                        if img_number == meta_img_num:
                            break
                    else:
                        try:
                            if bg_meta[self.bg_matching_par] == img_meta[self.bg_matching_par]:
                                break
                        except KeyError:
                            bg_file = None
                            continue

        if self.bg_type != 'Series Average':
            if bg_file is None:
                return 0.

            bg = np.asarray(read_image(bg_file), dtype=float)
            if bg is None or not np.isfinite(bg).any():
                return 0.

        if self.bg_scale != 1:
            bg *= self.bg_scale
            bg_scale_label = f'{self.bg_scale:0.2f} [Scale] x '
        if (self.bg_norm_channel != 'None') and (img_meta is not None) and (bg_meta is not None):
            try:
                if ((self.bg_norm_channel in img_meta.keys()) and
                        (self.bg_norm_channel in bg_meta.keys()) and
                        (bg_meta[self.bg_norm_channel] != 0)):
                    norm_factor = (img_meta[self.bg_norm_channel] / bg_meta[self.bg_norm_channel])
                    bg *= norm_factor
                    norm_label = f'{norm_factor:0.2f} [Normalized to Channel - {self.bg_norm_channel}] x '
            except (KeyError, TypeError):
                pass

        if self.bg_type != 'Series Average':
            self.sub_label = f'[Subtracted {bg_scale_label}{norm_label}{os.path.basename(bg_file)}]'
        else:
            self.sub_label = f'[Subtracted {bg_scale_label}{norm_label}{sname}]'

        return bg

    # ``save_1d`` moved to wranglerThread (the base class).
