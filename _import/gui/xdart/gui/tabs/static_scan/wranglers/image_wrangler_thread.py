# -*- coding: utf-8 -*-
"""
imageThread — worker thread for image_wrangler.

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
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# pyFAI / fabio / h5py
import fabio
import h5py

# Qt imports
from pyqtgraph import Qt

# Project imports
from xdart.modules.live import LiveFrame, LiveScan, IncidenceAngleUnresolved
from ssrl_xrd_tools.integrate.gid import gi_1d_output_axis_key
from ssrl_xrd_tools.integrate.calibration import poni_to_integrator, get_detector
from ssrl_xrd_tools.reduction import GIFreezeError
from ssrl_xrd_tools.io.image import read_image, count_frames
from ssrl_xrd_tools.io.export import write_xye
from ssrl_xrd_tools.io.nexus import find_nexus_image_dataset
from ssrl_xrd_tools.io.metadata import read_image_metadata
from xdart.utils import get_series_avg
from xdart.utils.h5pool import get_pool as _get_h5pool
from xdart.modules.reduction import (
    freeze_live_scan_gi_ranges,
    frame_from_live_frame,
    open_live_reduction_session,
    StandardPlanCache,
    reduce_live_frames,
    sync_live_scan_gi_settings,
)
from .qt_nexus_sink import QtNexusSink
from .wrangler_widget import wranglerThread

# Batch execution policy (PERF-4b/WS-X1).  "streaming" (default) routes the
# batch through one persistent ReductionSession + QtNexusSink (submit-per-frame,
# single writer thread, thumbnail parallelized in the worker) — proven on the
# 651-frame Eiger scan at 8 cores to be byte-identical to and >= the old chunked
# path (2D 32.6s vs 38.8s, XYE 23.7s vs 25.4s, 1D ~equal).  "chunked" is the old
# read-chunk -> integrate -> Phase-2-write path, kept one cycle as a fallback.
#
# Override without editing code: set XDART_BATCH_EXECUTION=chunked (or
# =streaming) in the environment before launching xdart.  Read once at import.
_BATCH_EXECUTION = os.environ.get("XDART_BATCH_EXECUTION", "streaming").strip().lower()
if _BATCH_EXECUTION not in ("chunked", "streaming"):
    logger.warning("XDART_BATCH_EXECUTION=%r is not 'chunked' or 'streaming'; "
                   "using 'streaming'.", _BATCH_EXECUTION)
    _BATCH_EXECUTION = "streaming"

# Live (non-batch) execution policy (PERF-4b/WS-X1 #3 — unify live onto the
# streaming sink).  "streaming" (DEFAULT as of the WS-X1 flip) routes a non-batch
# *reprocess* (Phase 1/2 collect loop) through the SAME persistent
# ReductionSession + QtNexusSink as batch — batch + reprocess share this write
# path (true-live keeps its serial one, below) — and the parallel pool
# pipelines I/O+compute (651-frame 2D reprocess ~30s vs ~60-76s serial; 1D ~21 vs
# ~27s; the live display is published GUI-side via _published_frames, see
# QtNexusSink._publish_display).  "serial" is the legacy per-frame _process_one
# reprocess path, kept one cycle as a fallback (XDART_LIVE_EXECUTION=serial).
# NOTE: the true-live *watch* (Phase 3, detector-rate, in-order one-at-a-time)
# always uses _process_one regardless of this flag — streaming's parallelism is
# moot there, and it stays the proven path.
_LIVE_EXECUTION = os.environ.get("XDART_LIVE_EXECUTION", "streaming").strip().lower()
if _LIVE_EXECUTION not in ("serial", "streaming"):
    logger.warning("XDART_LIVE_EXECUTION=%r is not 'serial' or 'streaming'; "
                   "using 'streaming'.", _LIVE_EXECUTION)
    _LIVE_EXECUTION = "streaming"


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _is_eiger_master(path):
    """Return True if path looks like an Eiger HDF5 master file (*_master.h5 / *_master.hdf5)."""
    return Path(path).stem.endswith('_master')


def _raw_lives_in_source(path):
    """Return True if the raw image for ``path`` is already embedded in the
    source file (Eiger master or any HDF5/NeXus container).

    When this is the case, writing ``map_raw`` into the output scan HDF5
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


def _gi_2d_range_keys(args):
    """Return the GI 2D range keys for the selected output mode."""
    mode = args.get('gi_mode_2d', 'qip_qoop')
    if mode == 'qip_qoop':
        return 'x_range', 'y_range'
    return 'radial_range', 'azimuth_range'


# RESTRUCTURE-TODO(B2): the GI-scout helpers below (_padded_axis_range,
# _result_intensity_all_dummy, _freeze_gi_2d_ranges_from_result,
# _freeze_gi_1d_range_from_result) plus the scout methods on imageThread
# (_scout_pending_frames, _build_scout, _freeze_gi_1d_auto_range,
# _freeze_gi_2d_auto_ranges) are TEST-ONLY.  Production GI common-grid freezing
# now lives in ssrl_xrd_tools.reduction.ReductionSession; these remain only so
# the GI live==batch==reload equivalence tests can validate the ssrl freeze
# against the original xdart scout (the tests bind the methods via MethodType).
# Relocate this whole cluster to a tests/ helper module together with that test
# refactor -- it should not sit in a production module.
def _padded_axis_range(axis, pad_fraction=0.02):
    """Return a slightly padded finite range for an integrated axis.

    Returns ``None`` when the axis is missing, has no finite samples, or is
    *collapsed* (span <= 0 — every finite value identical).  A collapsed
    axis means the scout integration was degenerate (e.g. GI at a 0°
    incidence); freezing a padded tiny range from it would clamp every
    subsequent frame onto that collapsed grid and blank the whole scan.
    Returning ``None`` leaves the range unfrozen so the caller can surface
    the problem instead of silently squashing the output.
    """
    if axis is None:
        return None
    arr = np.asarray(axis, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    span = hi - lo
    if span <= 0:
        return None
    pad = max(span * pad_fraction, 1e-9)
    return lo - pad, hi + pad


def _result_intensity_all_dummy(result, dummy=-1.0):
    """True if a 2D integration result has no real signal.

    A grazing-incidence map integrated at a degenerate incidence (e.g. a
    defaulted 0°) comes back as an all-dummy (``<= -1``) / empty grid — a
    blank cake.  The scout uses this to refuse freezing a representative
    grid off a blank scout frame, and tests use it to guard against the
    eiger "all -1.0" regression.
    """
    intensity = getattr(result, 'intensity', None)
    if intensity is None:
        return False
    arr = np.asarray(intensity, dtype=float)
    if arr.size == 0:
        return True
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return True
    return bool(np.all(finite <= dummy))


def _freeze_gi_2d_ranges_from_result(args, result):
    """Freeze missing GI 2D auto-range args from one scout result."""
    x_key, y_key = _gi_2d_range_keys(args)
    missing = [key for key in (x_key, y_key) if args.get(key) is None]
    if not missing:
        return False
    ranges = {
        x_key: _padded_axis_range(getattr(result, 'radial', None)),
        y_key: _padded_axis_range(getattr(result, 'azimuthal', None)),
    }
    changed = False
    for key in missing:
        if ranges[key] is not None:
            args[key] = ranges[key]
            changed = True
    return changed


def _gi_1d_output_range_key(gi_mode_1d):
    """Which integration range param controls the 1D *output* axis for a GI mode.

    Thin delegate to the canonical ssrl mapping
    (:func:`ssrl_xrd_tools.integrate.gid.gi_1d_output_axis_key`): ``azimuth_range``
    for q_oop/exit_angle (out-of-plane output), else ``radial_range``.  Freezing
    the wrong key leaves the output axis auto-ranging per incidence → a
    non-uniform stack the writer rejects."""
    return gi_1d_output_axis_key(gi_mode_1d)


def _freeze_gi_1d_range_from_result(args, result, gi_mode_1d=None):
    """Freeze the missing GI 1D *output-axis* range from one scout result so all
    frames share one axis.  Picks radial_range vs azimuth_range by mode (see
    :func:`_gi_1d_output_range_key`)."""
    key = _gi_1d_output_range_key(gi_mode_1d)
    if args.get(key) is not None:
        return False
    rng = _padded_axis_range(getattr(result, 'radial', None))
    if rng is None:
        return False
    args[key] = rng
    return True


# ---------------------------------------------------------------------------
# Natural sort helpers
# ---------------------------------------------------------------------------

# Pre-compiled regex patterns (avoids recompilation on every sort key call)
_INT_PATTERN = re.compile(r'(\d+)')
_FLOAT_PATTERN = re.compile(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)')

# Maximum number of frames to accumulate in the pending list before
# dispatching a partial batch.  This is the batch *chunk* size: read N →
# integrate N across workers → thumbnails → serial HDF5 write → repeat.
# Each barrier (read/integrate/write phase boundary) leaves cores idle, so
# fewer, bigger chunks amortise the ThreadPoolExecutor dispatch + the
# end-of-batch scan._save_to_nexus flush over more frames (PERF-4a).
#
# Peak buffer per batch ≈ flush_size × ~18 MB (one 2167x2070 int32 Eiger
# frame), plus the 2D cake per frame.  PERF-3 frees each frame's raw at
# end-of-batch, so this peak is now bounded to ONE chunk regardless of scan
# length (pre-PERF-3 the stash pinned raw for every frame for the whole
# scan), which is what makes raising the chunk size safe.  64 frames ≈
# ~1.5 GB peak — fine on a workstation; dial back toward 16 only on a
# RAM-starved / spinning-disk laptop (smaller reads finish sooner, so
# first-batch latency is lower).
_PENDING_FLUSH_SIZE = 64
# 1D-only batches carry no multi-MB cake arrays, so a larger pending buffer
# (fewer, bigger parallel dispatches) is cheaper without the 2D RAM cost.  Kept
# separate so the 2D-sized buffer is never globally raised.  256 × ~18 MB ≈
# ~4.6 GB peak (PERF-3 bounds it to one chunk).  (PERF-2 / PERF-4a)
_PENDING_FLUSH_SIZE_1D = 256

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
# — no need to duplicate `map_raw` into the output scan HDF5.  This is
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
# imageThread
# ---------------------------------------------------------------------------

class imageThread(wranglerThread):
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
        scan_args: dict, used as **kwargs in scan initialization.
            see LiveScan.
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
    # Pause: emitted (on the worker thread) AFTER _enter_pause has drained the
    # in-flight window + flushed the sink to .nxs at a frame boundary -- the GUI
    # slot then lifts the freeze guard, race-safely (writer is provably quiet).
    sigPaused = Qt.QtCore.Signal()

    def __init__(
            self,
            command_queue,
            scan_args,
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
            scan,
            data_1d,
            data_2d,
            live_mode=False,
            max_cores=1,
            parent=None):

        super().__init__(command_queue, scan_args, fname, file_lock, parent)

        self.h5_dir = h5_dir
        # Pause: the LiveScan currently being processed (a local in
        # process_scan), stashed so _enter_pause's serial branch can flush it.
        self._active_scan = None
        # N1: project root for portable @source_base; set from the wrangler in
        # setup() (None -> absolute raw paths, back-compat).
        self.source_base = None
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
        # Optional explicit SPEC search dir.  Set by the wrangler
        # widget (set_meta_dir / setup) — None / '' falls back to
        # the ssrl_xrd_tools default heuristic.  Threaded through
        # to every read_image_metadata call below.
        self.meta_dir = None
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
        # _frames_since_save / _published_frames are all initialised
        # by the wranglerThread base class — see wrangler_widget.py.
        # imageThread only needs to set the spec-specific extras here.
        self.max_cores = max_cores
        self.command = command
        self.scan = scan
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
        # Pause: start each run with a fresh serial-flush handle + save counter
        # so an early pause of THIS run can never flush a prior (stopped) run's
        # scan via stale state.
        self._active_scan = None
        self._frames_since_save = 0
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

        # Sync GI mode selections from image_wrangler into scan.bai_*_args
        if self.gi:
            self.scan.bai_1d_args['gi_mode_1d'] = self.gi_mode_1d
            self.scan.bai_2d_args['gi_mode_2d'] = self.gi_mode_2d

        try:
            self.process_scan()
        finally:
            self._close_reduction_session()
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
        if self.xye_only:
            # Int 1D (XYE) writes only .xye files (no .nxs) into <dir>/<scan>.
            logger.info('Output (XYE) folder: %s\n',
                        os.path.join(self.h5_dir, self.scan_name))
        else:
            output_path = getattr(self, 'fname', None) or getattr(self.scan, 'data_file', None)
            if output_path:
                logger.info('Output file: %s\n', output_path)

    def process_scan(self):
        """Batch-integrate all existing images, then optionally watch for new ones (live mode).

        Phase 1 — Collect: drain the current directory glob into a pending list.
        Phase 2 — Process: sequential with cached AzimuthalIntegrator (~0.35 s/frame).
        Phase 3 — Watch (live mode only): poll every 2 s for new files; process each immediately.
        """
        scan = None
        files_processed = 0
        _cached_poni = None
        is_eiger = _is_eiger_master(self.img_file) if self.img_file else False
        # One-time visibility into which execution path this run takes (so the
        # XDART_BATCH_EXECUTION / XDART_LIVE_EXECUTION flags are observable).
        # DEBUG: developer diagnostics, not run output.
        logger.debug('Execution policy: batch_mode=%s  batch=%s  live=%s',
                     self.batch_mode, self._batch_execution(), self._live_execution())

        # ── Phase 1 & 2: collect then process all existing images ─────────────
        pending = []  # [(img_file, img_number, img_data, img_meta, bg_raw)]
        # Per-flush read-time accumulator.  With the prefetcher this is mostly
        # queue-wait time on the main thread, not raw h5py I/O.
        _t_read_accum = 0.0

        while True:
            self._wait_if_paused()        # freeze here (before reading) if paused
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

            # Flush and switch scan when scan name changes.
            #
            # Two save paths matter here, both needed to keep N-1
            # .nxs files from ending up empty in multi-scan mode
            # (the bug observed with Eiger Image-Directory + LaB6
            # calibration scans, one frame per master):
            #
            # 1. ``force_save=True`` on the pending dispatch — covers
            #    batch_mode runs where the partial pending tail at
            #    scan boundary stayed in-memory because the serial
            #    path only writes when ``_frames_since_save >=
            #    LIVE_SAVE_INTERVAL``.
            # 2. Explicit ``_save_to_nexus`` on the old scan — covers
            #    non-batch (live-mode-style) runs where each frame
            #    was already dispatched immediately by the line-529
            #    cadence (``pending`` is empty by the time we get
            #    here), but ``_frames_since_save`` accumulated without
            #    hitting the save interval.  Without this, the old
            #    scan's in-memory frames get dropped when
            #    ``initialize_scan()`` reassigns the local variable.
            #
            # Only the end-of-loop final-flush passed ``force_save``
            # before, which is why only the last scan's file ever
            # got data.
            if (scan is None) or (scan_name != scan.name):
                if pending:
                    files_processed += self._dispatch_batch(
                        scan, pending, force_save=True,
                    )
                    pending = []
                # Catch the case where pending was already drained by
                # the per-iteration dispatch cadence below: the old
                # scan may have integrated-but-unsaved frames in
                # memory whose save_to_nexus call never fired.
                if (scan is not None
                        and not self.xye_only
                        and self._frames_since_save > 0):
                    _get_h5pool().pause(scan.data_file)
                    try:
                        with self.file_lock:
                            scan._save_to_nexus()
                    finally:
                        _get_h5pool().resume(scan.data_file)
                    self._flush_xye_buffer(scan)
                    logger.info(
                        '[SAVE-ON-SWAP] %d frames flushed for %s',
                        self._frames_since_save, scan.name,
                    )
                    self._frames_since_save = 0
                scan = self.initialize_scan()
                self._active_scan = scan      # Pause: serial-flush handle
                _cached_poni = None

            # Rebuild cached AzimuthalIntegrator when poni identity changes
            if self.poni is not _cached_poni:
                scan._cached_integrator = poni_to_integrator(self.poni)
                scan._cached_poni = self.poni
                scan._cached_fiber_integrator = None
                _cached_poni = self.poni
                self._cached_gi_incident_angle = None

            if img_number in scan.frames.index:
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
            # Batch mode accumulates a pending buffer (_PENDING_FLUSH_SIZE /
            # _PENDING_FLUSH_SIZE_1D) so the headless reduction spine can
            # integrate them in parallel.  Live (non-batch) mode dispatches
            # immediately so the GUI sees per-frame updates as soon as each
            # image lands; the disk save still gets batched separately inside
            # _dispatch_batch_serial via _frames_since_save.
            flush_size = (
                (_PENDING_FLUSH_SIZE_1D if getattr(self.scan, "skip_2d", False)
                 else _PENDING_FLUSH_SIZE)
                if self.batch_mode else 1
            )
            if len(pending) >= flush_size:
                if self.batch_mode:
                    self.showLabel.emit(
                        f'Integrating {len(pending)} frames (partial batch)...'
                    )
                _t_disp = time.time()
                files_processed += self._dispatch_batch(scan, pending)
                if self.batch_mode or _t_read_accum >= 0.5:
                    logger.info(
                        '[FLUSH] %d frames  read=%.2fs  dispatch=%.2fs',
                        len(pending), _t_read_accum, time.time() - _t_disp,
                    )
                pending = []
                _t_read_accum = 0.0

        # Process whatever is left.  force_save=True so any remaining
        # _frames_since_save tail in live mode is flushed to disk.
        if pending and scan is not None and self.command != 'stop':
            _t_disp = time.time()
            files_processed += self._dispatch_batch(
                scan, pending, force_save=True,
            )
            logger.info(
                '[FLUSH-FINAL] %d frames  read=%.2fs  dispatch=%.2fs',
                len(pending), _t_read_accum, time.time() - _t_disp,
            )
        elif (scan is not None and not self.xye_only
              and self._frames_since_save > 0 and self.command != 'stop'):
            # Pending was empty but the live-save batcher has unflushed
            # frames (last batch hit the divisor exactly).  Force a save
            # before leaving the collect loop.
            _get_h5pool().pause(scan.data_file)
            try:
                with self.file_lock:
                    scan._save_to_nexus()
            finally:
                _get_h5pool().resume(scan.data_file)
            self._flush_xye_buffer(scan)
            self._frames_since_save = 0

        # ── Phase 3: live watching ────────────────────────────────────────────
        if self.live_mode and self.command != 'stop' and scan is not None:
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
                self._wait_if_paused()    # pause/examine/resume a live acquisition
                if self.command == 'stop':
                    break                 # Pause -> Stop while watching
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

                if (scan is None) or (scan_name != scan.name):
                    scan = self.initialize_scan()
                    self._active_scan = scan  # Pause: serial-flush handle
                    _cached_poni = None

                if self.poni is not _cached_poni:
                    scan._cached_integrator = poni_to_integrator(self.poni)
                    scan._cached_poni = self.poni
                    scan._cached_fiber_integrator = None
                    _cached_poni = self.poni
                    self._cached_gi_incident_angle = None

                if img_number in scan.frames.index:
                    continue

                bg_raw = self.get_background(img_file, img_number, img_meta)
                # Process immediately — single-threaded for low latency.
                # _process_one uses add_frame(batch_save=True), so the
                # disk save still rides on _frames_since_save below.
                self._process_one(scan, img_file, img_number, img_data, img_meta, bg_raw)
                files_processed += 1
                self._frames_since_save += 1
                if self._save_due(scan):
                    _get_h5pool().pause(scan.data_file)
                    try:
                        with self.file_lock:
                            scan._save_to_nexus()
                    finally:
                        _get_h5pool().resume(scan.data_file)
                    self._flush_xye_buffer(scan)
                    self._frames_since_save = 0

        # Final flush on exit (live-watch tail or stop request) so the
        # last few frames aren't lost.
        if (scan is not None and not self.xye_only
                and self._frames_since_save > 0):
            _get_h5pool().pause(scan.data_file)
            try:
                with self.file_lock:
                    scan._save_to_nexus()
            finally:
                _get_h5pool().resume(scan.data_file)
            self._flush_xye_buffer(scan)
            self._frames_since_save = 0

        # In batch mode, emit a single final signal so the GUI can refresh
        if self.batch_mode and files_processed > 0:
            self.sigUpdate.emit(-1)
        logger.info('Total Files Processed: %d', files_processed)

    # ── Batch dispatch ────────────────────────────────────────────────────────

    def _dispatch_batch(self, scan, pending, *, force_save=False):
        """Process a list of pending images — parallel in batch mode, serial otherwise.

        ``force_save`` only affects the serial path (live mode); the
        parallel path always saves at end of batch by construction
        (the whole point of batch mode is one big save per dispatch).
        """
        self._maybe_warn_live_gi_clip()
        if self.batch_mode and len(pending) > 1:
            if self._batch_execution() == "streaming":
                return self._dispatch_batch_streaming(scan, pending)
            return self._dispatch_batch_parallel(scan, pending)
        # Live (non-batch): #3 routes it through the SAME streaming session +
        # QtNexusSink (which does the per-frame display publish for live), behind
        # the live flag; the proven per-frame _process_one path is the default.
        if self._live_execution() == "streaming":
            return self._dispatch_batch_streaming(scan, pending)
        return self._dispatch_batch_serial(scan, pending, force_save=force_save)

    def _batch_execution(self) -> str:
        """Active batch execution policy ('chunked' | 'streaming').  Instance
        override (``self.batch_execution``) wins over the module default."""
        return getattr(self, "batch_execution", None) or _BATCH_EXECUTION

    def _live_execution(self) -> str:
        """Active live (non-batch) execution policy ('serial' | 'streaming').
        Instance override (``self.live_execution``) wins over the module default."""
        return getattr(self, "live_execution", None) or _LIVE_EXECUTION

    def _maybe_warn_live_gi_clip(self) -> None:
        """One-time advisory for live GI runs (#75).

        In a live run the common GI output grid is frozen from the FIRST frame
        only — there's no lookahead to later incidence angles — so a scan that
        sweeps incidence can write clipped later frames (uniform axis, no crash,
        truncated tail).  Batch reprocessing uses the union of the incidence
        extremes (#70) and recovers the full range.  Phrased conditionally
        ("if this scan sweeps …") because live can't yet know whether incidence
        varies, and source-agnostic (no assumption about the frame source).
        Batch runs bracket all frames, so this never fires there.
        """
        if (not self.gi or self.batch_mode
                or getattr(self, '_warned_live_gi_clip', False)):
            return
        self._warned_live_gi_clip = True
        msg = ('Live GI: output range frozen from the first frame — if this '
               'scan sweeps a range of incidence angles, later frames may be '
               'clipped. Reprocess in batch for the full range.')
        logger.warning(msg)
        try:
            self.showLabel.emit(msg)
        except Exception:
            logger.debug("showLabel emit failed for live-GI clip warning",
                         exc_info=True)

    # ``_resolve_frame_mask``, ``_apply_threshold_inline``, and
    # ``_flush_xye_buffer`` moved to wranglerThread (the base class)
    # in May 2026 — both imageThread and nexusThread inherit them now.
    # See xdart/gui/tabs/static_scan/wranglers/wrangler_widget.py.

    # RESTRUCTURE-TODO(B2): _scout_pending_frames / _build_scout /
    # _freeze_gi_1d_auto_range / _freeze_gi_2d_auto_ranges are test-only scout
    # fixtures (see the cluster note near _padded_axis_range) -- relocate to a
    # tests/ helper module with the GI-equivalence test refactor.
    def _scout_pending_frames(self, pending):
        """Return bounded representative pending entries for freeze tests.

        Production no longer calls this pre-pass; GI common-grid freezing is
        driven by :class:`ssrl_xrd_tools.reduction.ReductionSession` when the
        reduction session opens.  The method remains as a compatibility fixture
        for real-data tests that inspect scout selection directly.
        """
        if len(pending) <= 1:
            return list(pending)
        motor = self.incidence_motor
        try:
            float(motor)
            return [pending[0]]
        except (TypeError, ValueError):
            pass
        resolved = []
        any_unresolved = False
        for i, entry in enumerate(pending):
            meta = entry[3] or {}
            try:
                resolved.append((i, float(meta.get(motor))))
            except (TypeError, ValueError):
                any_unresolved = True
        idxs = set()
        if resolved:
            idxs.add(min(resolved, key=lambda item: item[1])[0])
            idxs.add(max(resolved, key=lambda item: item[1])[0])
        if any_unresolved or not resolved:
            idxs.update((0, len(pending) - 1))
        return [pending[i] for i in sorted(idxs)]

    def _build_scout(self, scan, entry):
        """Build a temporary ``LiveFrame`` for the headless freeze adapter."""
        img_file, img_number, img_data, img_meta, bg_raw, _ = entry
        img_data = self._apply_threshold_inline(img_data)
        frame_mask = self._resolve_frame_mask(scan, img_data)
        scratch = LiveFrame(
            img_number, img_data, poni=self.poni,
            scan_info=img_meta, static=True, gi=True,
            th_mtr=self.incidence_motor, bg_raw=bg_raw,
            sample_orientation=self.sample_orientation,
            tilt_angle=self.tilt_angle,
            series_average=self.series_average,
            integrator=scan._cached_integrator,
            mask=frame_mask,
        )
        if img_file:
            scratch.source_file = os.path.abspath(str(img_file))
        if _raw_lives_in_source(img_file):
            scratch.source_frame_idx = int(img_number) - 1
        else:
            scratch.source_frame_idx = 0
        scratch._get_incident_angle()
        return scratch

    def _freeze_gi_1d_auto_range(self, scan, pending) -> None:
        """Compatibility wrapper for tests; delegates freeze to ssrl."""
        if not self.gi or not pending:
            return
        args = getattr(scan, 'bai_1d_args', None)
        if not isinstance(args, dict):
            return
        key = gi_1d_output_axis_key(args.get('gi_mode_1d', 'q_total'))
        if args.get(key) is not None:
            return
        sync_live_scan_gi_settings(
            scan,
            incidence_motor=self.incidence_motor,
            sample_orientation=self.sample_orientation,
            tilt_angle=self.tilt_angle,
        )
        scouts = [self._build_scout(scan, entry)
                  for entry in self._scout_pending_frames(pending)]
        freeze_live_scan_gi_ranges(
            scan,
            scouts,
            scan_name=str(getattr(scan, "name", "scan")),
            global_mask=self.mask,
            integrator=getattr(scan, "_cached_integrator", None),
            poni=self.poni,
            integrate_1d=True,
            integrate_2d=False,
            gi_freeze_mode="scout_union" if self.batch_mode else "first_frame",
        )

    def _freeze_gi_2d_auto_ranges(self, scan, pending) -> None:
        """Compatibility wrapper for tests; delegates freeze to ssrl."""
        if (not self.gi or self.xye_only or getattr(scan, 'skip_2d', False)
                or not pending):
            return
        args = getattr(scan, 'bai_2d_args', None)
        if not isinstance(args, dict):
            return
        keys = _gi_2d_range_keys(args)
        if all(args.get(key) is not None for key in keys):
            return
        sync_live_scan_gi_settings(
            scan,
            incidence_motor=self.incidence_motor,
            sample_orientation=self.sample_orientation,
            tilt_angle=self.tilt_angle,
        )
        scouts = [self._build_scout(scan, entry)
                  for entry in self._scout_pending_frames(pending)]
        freeze_live_scan_gi_ranges(
            scan,
            scouts,
            scan_name=str(getattr(scan, "name", "scan")),
            global_mask=self.mask,
            integrator=getattr(scan, "_cached_integrator", None),
            poni=self.poni,
            integrate_1d=False,
            integrate_2d=True,
            gi_freeze_mode="scout_union" if self.batch_mode else "first_frame",
        )

    def _gi_ranges_fully_pinned(self, scan) -> bool:
        """True when every GI output range this run would auto-freeze is
        already explicitly set (T0-3).

        Mirrors the self-skip conditions of :meth:`_freeze_gi_1d_auto_range`
        (the active 1D mode's output-axis range key is set) and
        :meth:`_freeze_gi_2d_auto_ranges` (both of the active 2D mode's range
        keys are set; 2D irrelevant for XYE-only / skip_2d runs).  When all
        relevant ranges are pinned there is no auto grid to freeze, so the
        whole-scan scout — and its fail-closed abort on unverifiable sources —
        is unnecessary.  Conservative: any non-dict args → False (let the
        normal prepass decide)."""
        args_1d = getattr(scan, 'bai_1d_args', None)
        if not isinstance(args_1d, dict):
            return False
        key = gi_1d_output_axis_key(args_1d.get('gi_mode_1d', 'q_total'))
        if args_1d.get(key) is None:
            return False
        if self.xye_only or getattr(scan, 'skip_2d', False):
            return True
        args_2d = getattr(scan, 'bai_2d_args', None)
        if not isinstance(args_2d, dict):
            return False
        return all(args_2d.get(k) is not None
                   for k in _gi_2d_range_keys(args_2d))

    # ── BLOCKER 1: whole-scan GI grid freeze (streaming batch) ──────────────
    def _gi_freeze_whole_scan_prepass(self, scan) -> bool:
        """Freeze the GI common q/χ grid from the WHOLE scan's incidence range
        BEFORE the streaming session opens.

        The streaming batch session is built from the FIRST chunk only, so its
        ``scout_union`` would bracket chunk 1's incidence range and clip later,
        higher-incidence frames (BLOCKER 1).  Here a CHEAP metadata-only sweep
        finds the global lowest+highest-incidence frames, loads ONLY those two
        images, and runs the existing whole-scan freeze
        (:meth:`_freeze_gi_1d_auto_range` / :meth:`_freeze_gi_2d_auto_ranges`,
        which delegate to ``freeze_live_scan_gi_ranges``) to write the UNION
        ranges into ``scan.bai_*_args``.  The streaming session then rebuilds its
        plan with those ranges (``StandardPlanCache`` keys on ``bai_*_args``) and
        its own per-session freeze early-returns -> the whole batch shares the
        union grid.  The two scouts are integrated only inside the throwaway
        freeze session, never submitted to the streaming session -> no double
        processing.

        Batch + streaming + GI only.  True-live keeps ``first_frame`` serial (it
        has no whole scan ahead of time); the chunked batch path already builds
        the full pending up front so its first/last ARE the scan extremes.
        Runs once per scan.  Eiger single-master (one incidence per master) and
        fixed/manual incidence collapse to the existing behaviour (no whole-scan
        scout) -- the grid was never chunk-clipped there.

        **Policy (T0-4, Vivek 2026-06-09): warn-and-proceed.**  When the
        whole-scan incidence range cannot be established (unverifiable source,
        unreadable metadata, scout failure), the run PROCEEDS on the session's
        own first-chunk freeze with a one-time user-visible advisory, instead
        of aborting.  Rationale: frames are binned natively onto the common
        grid (values inside the range are exact — no interpolation), and at
        the beamline incidence varies too little for the cropped extreme tails
        to matter; the union sweep is kept where it is free (Image-Series with
        readable metadata).  The ONLY fail-closed exit left is a freeze that
        actually errors (e.g. a degenerate scout cake) — proceeding there
        would integrate onto a broken grid, not a slightly narrow one.

        Returns ``True`` when the caller may open the session, ``False`` when
        the run was aborted (freeze error only; caller must not proceed).
        """
        if not (self.gi and self.batch_mode
                and self._batch_execution() == "streaming"):
            return True
        if getattr(self, "_gi_prepass_scan_id", None) == id(scan):
            return True
        if self._gi_ranges_fully_pinned(scan):
            # T0-3: every GI output range this run would auto-freeze is already
            # explicitly set — the freeze functions self-skip and the session's
            # own per-session freeze early-returns, so there is no auto grid to
            # chunk-clip and no scout sweep is needed.  Without this pre-flight
            # the "unverifiable" abort below fired even for fully-pinned runs
            # (and its "set fixed/manual GI ranges" remedy didn't work).
            self._gi_prepass_scan_id = id(scan)   # latch: decided for this scan
            return True
        try:
            status, scouts = self._gi_whole_scan_scout_entries(scan)
        except Exception as exc:
            # T0-4 warn-and-proceed: a scout failure means we lose the union
            # sweep, not correctness — the first-chunk freeze is acceptable.
            self._warn_gi_first_chunk_freeze(
                f"scout pre-pass raised ({exc})", scan)
            self._gi_prepass_scan_id = id(scan)
            return True
        if status == "abort":
            # Sweep ran but couldn't read enough incidences from metadata.
            self._warn_gi_first_chunk_freeze(
                "could not establish the whole-scan incidence range from "
                "frame metadata", scan)
            self._gi_prepass_scan_id = id(scan)
            return True
        if status == "unverifiable":
            # Image-Directory or named-motor Eiger: per-frame incidence can't
            # be cheaply swept up front (for Eiger it lives in the SPEC
            # sidecar, per frame).  T0-4 policy: proceed on the first-chunk
            # freeze with an advisory — values inside the grid are exact; only
            # extreme-incidence tails beyond it are cropped.
            source = (getattr(self, "inp_type", None) or
                      ("Eiger master" if (getattr(self, "img_file", "") and
                                          _is_eiger_master(getattr(self, "img_file", "")))
                       else "this source"))
            self._warn_gi_first_chunk_freeze(
                f"the whole-scan incidence range cannot be swept up front for "
                f"'{source}'", scan)
            self._gi_prepass_scan_id = id(scan)
            return True
        if status == "freeze":
            # These self-skip when the relevant ranges are already set, and
            # freeze the UNION over the two extreme scouts we hand them (NOT a
            # chunk).  A degenerate scout cake raises GIFreezeError here --
            # that stays FAIL-CLOSED via _abort_gi_prepass (the freeze ran and
            # produced a broken grid, not a narrow one) and must not escape
            # the worker thread (run() has no except).
            try:
                self._freeze_gi_1d_auto_range(scan, scouts)
                self._freeze_gi_2d_auto_ranges(scan, scouts)
            except Exception as exc:
                self._abort_gi_prepass(f"whole-scan grid freeze failed ({exc})")
                return False
        # status == "skip": fixed/manual/single-incidence/Eiger -- the session's
        # own freeze is correct (the grid was never chunk-clipped there).
        self._gi_prepass_scan_id = id(scan)   # latch only after success
        return True

    def _warn_gi_first_chunk_freeze(self, reason: str, scan=None) -> None:
        """T0-4 warn-and-proceed advisory: the GI output grid will be frozen
        from the first frames instead of the whole-scan incidence union.
        Values inside the grid are exact (frames bin natively onto it); only
        extreme-incidence tails beyond it are cropped — accepted at the
        beamline (incidence varies too little to matter).  One per scan: the
        prepass latches the scan id right after this fires.

        When ``scan`` is given the advisory is also stamped onto it
        (``gi_freeze_diagnostic``) so the writer persists it in
        ``/entry/reduction/config`` — the disclosure survives in the output
        file, not just as a transient GUI label."""
        msg = (
            'GI: ' + reason + ' — output grid will be frozen from the first '
            'frames; extreme-incidence tails beyond it are cropped (values '
            'inside are exact). Set explicit 1D/2D output ranges for full '
            'coverage.')
        logger.warning(msg)
        if scan is not None:
            try:
                scan.gi_freeze_diagnostic = msg
            except Exception:
                logger.debug("could not stamp gi_freeze_diagnostic",
                             exc_info=True)
        try:
            self.showLabel.emit(msg)
        except Exception:
            logger.debug("showLabel emit failed for GI first-chunk-freeze "
                         "advisory", exc_info=True)

    def _abort_gi_prepass(self, reason: str) -> None:
        """Fail-closed exit for a GI freeze that actually ERRORED (degenerate
        scout cake etc.): surface a user-visible error and STOP the streaming
        GI run rather than integrate onto a broken grid.  (Unverifiable /
        unreadable-metadata cases warn-and-proceed instead — see
        _warn_gi_first_chunk_freeze.)"""
        msg = (
            'GI batch aborted: cannot freeze a whole-scan q/χ grid (' + reason +
            '). Refusing to integrate onto a partial grid that would clip later '
            'frames. Set Theta Motor to Manual and enter the incident angle, or '
            'check the per-frame metadata, then restart.')
        logger.error(msg)
        try:
            self.showLabel.emit(msg)
        except Exception:
            logger.debug("showLabel emit failed for GI prepass abort",
                         exc_info=True)
        # Stop the collection loop loudly (mirrors a write-failure stop); the
        # close path then drains/closes any partially-built session.  Under
        # command_lock so a concurrent GUI pause() can't overwrite it (RS-2).
        # getattr: tests drive this on duck holders without the lock.
        _lock = getattr(self, "command_lock", None)
        if _lock is not None:
            with _lock:
                self.command = 'stop'
        else:
            self.command = 'stop'

    def _gi_whole_scan_scout_entries(self, scan):
        """Decide how to freeze the whole-scan GI grid and, when needed, gather
        the scout images.  Returns ``(status, entries)``:

        - ``("skip", [])`` — fixed/manual angle, Eiger/master, Image-Directory,
          a single-frame scan, or a swept scan with one incidence: the existing
          chunk-local / session freeze is correct (the grid was never clipped).
        - ``("freeze", [lo_entry, hi_entry])`` — a varying-incidence per-file
          scan whose extremes were resolved from readable metadata; ``entries``
          are ``(img_file, img_number, img_data, img_meta, bg_raw, 0.0)`` for the
          global lowest+highest incidence frames, with their images loaded.
        - ``("abort", [])`` — a multi-file scan whose incidence we genuinely
          cannot establish (fewer than two readable incidences across the whole
          series).  The caller warns and proceeds on the first-chunk freeze
          (T0-4 policy; the status name predates it).

        Robust to per-frame metadata GAPS: it finds the extremes from the
        readable frames and only aborts when it can't read enough to establish a
        global range at all (BLOCKER 1 follow-up)."""
        motor = self.incidence_motor
        try:
            float(motor)        # fixed/manual: one angle for the whole scan
            return "skip", []
        except (TypeError, ValueError):
            pass
        files = self._enumerate_scan_files()
        if not files:
            # _enumerate_scan_files() returns [] for sources that can't be cheaply
            # swept per-frame.  Distinguish the safe cases (skip) from the risky
            # ones (unverifiable = we can't confirm the whole-scan range, so the
            # caller must fail closed rather than silently clip later frames):
            #
            # SAFE ("skip"):
            #   - Eiger SINGLE-master (one incidence per master → fixed grid, the
            #     session's own freeze is correct and was never chunk-clipped).
            #   - Any scan whose source-host attrs are absent/incomplete (no file
            #     to check → missing-source host, not a TIFF angle-dependence run).
            # RISKY ("unverifiable"):
            #   - Image-Directory per-file GI scan (inp_type == 'Image Directory'):
            #     a TIFF angle-dependence sweep that _enumerate_scan_files() skips.
            #     The session WOULD clip later higher-incidence frames.
            #   - Eiger master when the motor is not fixed/manual (if multi-master
            #     Eiger is possible: each master a different incidence → same clip
            #     risk; we can't distinguish single- vs multi-master before the run).
            #
            # For "unverifiable" the caller WARNS and proceeds on the
            # first-chunk freeze (T0-4 policy: cropped extreme tails are
            # accepted; values inside the grid are exact).  Source-aware
            # whole-scan incidence enumeration remains a possible refinement,
            # demoted from correctness work to nice-to-have.
            img_file = getattr(self, "img_file", None) or ""
            if getattr(self, "inp_type", None) == "Image Directory":
                # Per-file angle-dependence loaded as a directory: can't sweep.
                return "unverifiable", []
            if img_file and _is_eiger_master(img_file):
                # Eiger master: single-master (fixed angle per master) is safe;
                # multi-master (varying angle) is risky but indistinguishable
                # here.  Conservatively treat any non-fixed-motor Eiger as
                # unverifiable so a multi-master angle-dependence can't clip.
                return "unverifiable", []
            # Missing source attrs / no img_file: can't be an angle-dependence sweep.
            return "skip", []
        if len(files) < 2:
            return "skip", []   # single-frame scan: no incidence range to freeze
        resolved = []
        for fname, img_number in files:
            try:
                meta = (read_image_metadata(fname, meta_format=self.meta_ext,
                                            meta_dir=self.meta_dir)
                        if self.meta_ext else {})
            except Exception:
                continue            # tolerate a per-frame metadata gap
            ang = self._resolve_incidence_from_meta(meta, motor)
            if ang is not None:
                resolved.append((ang, fname, img_number, meta))
        if len(resolved) < 2:
            # A multi-file (≥2) scan whose incidence we can't read for at least
            # two frames: we cannot establish the global range -> fail CLOSED.
            return "abort", []
        lo = min(resolved, key=lambda r: r[0])
        hi = max(resolved, key=lambda r: r[0])
        if lo[0] == hi[0]:          # no incidence sweep -> chunk grid is fine
            return "skip", []
        entries = []
        for _ang, fname, img_number, meta in (lo, hi):
            data = np.asarray(read_image(fname), dtype=float)
            # bg is irrelevant to the frozen AXIS extent (geometry+incidence
            # driven), so a failing/missing background must NOT abort an
            # otherwise-valid GI run -- degrade to 0 for the axis-only scout
            # rather than let the prepass turn it into a fail-closed stop.
            try:
                bg = self.get_background(fname, img_number, meta)
            except Exception:
                logger.debug("GI scout background failed for %s; using 0 for the "
                             "axis-only scout", fname, exc_info=True)
                bg = 0.0
            entries.append((fname, img_number, data, meta, bg, 0.0))
        return "freeze", entries

    @staticmethod
    def _resolve_incidence_from_meta(meta, motor):
        """Case-insensitive lookup of the incidence motor in a frame's metadata
        -> float, or None if absent/non-numeric (mirrors LiveFrame._get_incident_angle)."""
        if not isinstance(meta, dict):
            return None
        ml = str(motor).lower()
        for key, val in meta.items():
            if str(key).lower() == ml:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return None
        return None

    def _enumerate_scan_files(self):
        """Sorted ``(fname, img_number)`` for a per-file scan series, metadata
        only (NO image read).  Returns ``[]`` for Eiger/HDF5 masters and
        Image-Directory sources that can't be cheaply per-frame swept (and,
        defensively, for any host missing the source attrs)."""
        img_file = getattr(self, "img_file", None)
        if not img_file or _is_eiger_master(img_file):
            return []
        img_ext = getattr(self, "img_ext", "") or ""
        if img_ext.lower() in ('h5', 'hdf5', 'nxs'):
            return []
        if getattr(self, "inp_type", None) == 'Image Directory':
            return []
        scan_name = getattr(self, "scan_name", None)
        img_dir = getattr(self, "img_dir", None)
        if not scan_name or not img_dir:
            return []
        _series_re = re.compile(
            rf'^{re.escape(scan_name)}_\d+\.{re.escape(img_ext)}$')
        paths = [str(p) for p in Path(img_dir).glob(
                     f'{scan_name}_*.{img_ext}')
                 if _series_re.match(p.name)]
        out = []
        for fname in natural_sort_ints(paths):
            _sname, snumber = _get_scan_info(fname)
            out.append((fname, snumber))
        return out

    # ── Pause / Resume ──────────────────────────────────────────────────────
    def _wait_if_paused(self) -> None:
        """Block while a Pause is in effect, WITHOUT tearing down the scan or
        session.  Pause is a THIRD command state between the run state
        (``'start'``) and ``'stop'`` (the loops treat everything != 'stop' as go,
        so 'pause' must be special-cased here).

        On the first entry to pause, :meth:`_enter_pause` brings the file to a
        frame-boundary idle state and emits ``sigPaused`` so the GUI lifts the
        freeze guard for browsing; then we spin until ``command`` leaves
        ``'pause'`` — either ``'start'`` (resume: the caller continues its loop on
        the same open session) or ``'stop'`` (the caller's existing stop-check
        breaks next).  Call at the TOP of every processing loop, ABOVE the
        stop-check, so nothing new is read/submitted while paused.  The
        ``while`` exits on stop, so this is shutdown-safe (close() sets 'stop').
        """
        if self.command != 'pause':
            return
        self._enter_pause()
        while self.command == 'pause':
            time.sleep(0.05)

    #: Bound the pause drain so a hung pool worker (stalled IO / runaway pyFAI)
    #: can't deadlock the pause; the wait also bails early on Stop/close.
    PAUSE_DRAIN_TIMEOUT = 30.0

    def _enter_pause(self) -> None:
        """Quiesce the writer at a frame boundary so a paused user can browse any
        already-processed frame from disk, then signal the GUI to lift the guard.

        Routing is keyed on the SERIAL tail first: ``_frames_since_save > 0`` is
        the unambiguous signal that the serial path is the active writer (the
        true-live watch loop uses ``_process_one`` and increments it; the
        streaming path never does).  So during Phase-3 live watching — where the
        Phase-2 streaming session is still open but DORMANT — the serial flush
        correctly wins and the watch tail (incl. its XYE buffer) is persisted and
        the counter reset.  Otherwise the streaming branch drains+flushes.

        STRICT ordering: drain/flush MUST complete BEFORE ``sigPaused`` so the
        writer is provably idle before the GUI reads disk (race-safe).  The drain
        is bounded (:attr:`PAUSE_DRAIN_TIMEOUT`) + cancel-aware so a stuck worker
        can't strand the pause or block Stop/close.
        """
        try:
            session = getattr(self, '_streaming_session', None)
            sink = getattr(self, '_streaming_sink', None)
            scan = self._active_scan
            # Drain any open streaming session first (a no-op when nothing is
            # in-flight) so the writer is provably idle before we touch the
            # file from THIS thread.
            drained = True
            if session is not None:
                drained = session.drain(
                    timeout=getattr(self, 'PAUSE_DRAIN_TIMEOUT', 30.0))
            if not drained:
                # RS-1: the writer is provably NOT idle (a stalled worker) — a
                # save/flush from this thread would race the writer's own
                # write()→_flush() (single-writer invariant), and resetting the
                # save counter without saving would break persist-before-evict.
                # Pause still proceeds (submits have stopped); the tail flushes
                # on resume/finish.
                logger.warning("pause drain timed out (worker not idle); "
                               "pausing without a flush")
            elif scan is not None and self._frames_since_save > 0:
                # Serial path is the active writer (watch loop / serial
                # dispatch): flush the serial tail.
                if not self.xye_only:
                    _get_h5pool().pause(scan.data_file)
                    try:
                        with self.file_lock:
                            scan._save_to_nexus()
                    finally:
                        _get_h5pool().resume(scan.data_file)
                self._flush_xye_buffer(scan)
                self._frames_since_save = 0
            elif session is not None and sink is not None:
                # Streaming path (batch / reprocess / live Phase 2): in-flight
                # window drained (non-terminal — session stays open) + flush.
                sink._flush(force=True)
        except Exception:
            # A drain/flush failure must not strand the run; log loudly and
            # still signal the pause (we've stopped submitting, so the writer is
            # idle and disk reads are safe even if the final flush lagged).
            logger.error("error draining/flushing on pause", exc_info=True)
        finally:
            self.sigPaused.emit()

    def _save_due(self, scan, *, force=False):
        """Whether a non-batch v2 save should fire now (persist-before-evict).

        A save is due when forced (final flush), when ``LIVE_SAVE_INTERVAL``
        frames have accumulated since the last save, OR when the *unsaved*
        in-memory frame set is about to reach the frame-cache cap
        (``LiveFrameSeries._in_memory_cap``).  The cap bound is the data-loss
        fix: the writer reads int_1d/int_2d straight off the in-memory frames,
        and ``stash`` refuses to evict unsaved ones — so we must save before
        the unsaved set fills the cache, else it would grow unbounded.  Net:
        ``LIVE_SAVE_INTERVAL`` is an UPPER bound on save spacing; the cap is a
        HARD bound, so the high interval is safe even on scans longer than the
        cap.  Never saves in xye_only mode (no .nxs target).
        """
        if self.xye_only or self._frames_since_save <= 0:
            return False
        if force or self._frames_since_save >= self.LIVE_SAVE_INTERVAL:
            return True
        cap = getattr(scan.frames, "_in_memory_cap", 64)
        counter = getattr(scan.frames, "unsaved_in_memory_count", None)
        unsaved = counter() if callable(counter) else self._frames_since_save
        return unsaved >= max(1, cap - 8)

    def _dispatch_batch_serial(self, scan, pending, *, force_save=False):
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
            self._wait_if_paused()        # pause between frames (serial path)
            if self.command == 'stop':
                break
            # item is (img_file, img_number, img_data, img_meta, bg_raw, t_read)
            self._process_one(scan, *item)
            count += 1

        self._frames_since_save += count

        if self._save_due(scan, force=force_save):
            _t_save0 = time.time()
            _get_h5pool().pause(scan.data_file)
            try:
                with self.file_lock:
                    scan._save_to_nexus()
            finally:
                _get_h5pool().resume(scan.data_file)
            _t_save = time.time() - _t_save0
            _t_xye0 = time.time()
            self._flush_xye_buffer(scan)
            _t_xye = time.time() - _t_xye0
            logger.info(
                '[SAVE] %d frames since last save  save=%.3fs  xye=%.3fs',
                self._frames_since_save, _t_save, _t_xye,
            )
            self._frames_since_save = 0
        return count

    def _build_batch_frames(self, scan, pending):
        """Build the LiveFrame shells for a batch chunk (shared by the chunked
        and streaming dispatchers).  Stamps source refs + skip_map_raw."""
        skip_2d = scan.skip_2d
        gi = self.gi
        th_mtr = self.incidence_motor
        sample_orientation = self.sample_orientation
        tilt_angle = self.tilt_angle
        series_average = self.series_average
        frames = []
        for img_file, img_number, img_data, img_meta, bg_raw, _t_read in pending:
            if self.command == 'stop':
                break
            img_data = self._apply_threshold_inline(img_data)
            frame_mask = self._resolve_frame_mask(scan, img_data)
            frame = LiveFrame(
                img_number, img_data, poni=self.poni,
                scan_info=img_meta, static=True, gi=gi,
                th_mtr=th_mtr, bg_raw=bg_raw,
                sample_orientation=sample_orientation,
                tilt_angle=tilt_angle,
                series_average=series_average,
                integrator=scan._cached_integrator,
                mask=frame_mask,
            )
            if img_file:
                frame.source_file = os.path.abspath(str(img_file))
            else:
                frame.source_file = ""
            if _raw_lives_in_source(img_file):
                frame.source_frame_idx = int(img_number) - 1
            else:
                frame.source_frame_idx = 0
            frame.skip_map_raw = skip_2d or _raw_lives_in_source(img_file)
            frames.append(frame)
        return frames

    def _dispatch_batch_streaming(self, scan, pending):
        """Stream a batch chunk through a persistent ReductionSession +
        QtNexusSink (PERF-4b/WS-X1).

        One session spans the whole scan: it owns the worker pool + the single
        writer thread, and ``QtNexusSink`` owns the .nxs/XYE write.  Each frame
        is registered + submitted the instant it's built; the per-scan
        ``_close_reduction_session`` ``finish()`` drains the writer and does the
        final flush.  Behind the execution flag; chunked is the default.
        """
        if getattr(scan, "_cached_integrator", None) is None:
            logger.info('[STREAM] no cached integrator yet — first frame falls '
                        'back to serial (streaming engages from the next frame)')
            return self._dispatch_batch_serial(scan, pending)
        if getattr(scan, '_cached_data_mask', None) is None and pending:
            self._prewarm_frame_mask(scan, pending[0][2])
        sync_live_scan_gi_settings(
            scan,
            incidence_motor=self.incidence_motor,
            sample_orientation=self.sample_orientation,
            tilt_angle=self.tilt_angle,
        )
        # BLOCKER 1: freeze the GI common grid from the WHOLE scan's incidence
        # range before the session opens (no-op on chunks 2..N via the scan-id
        # guard; no-op for non-GI / fixed-incidence sources).  T0-4 policy:
        # an unverifiable/unestablishable range WARNS and proceeds on the
        # first-chunk freeze (diagnostic persisted in provenance); only a
        # freeze that actually errors aborts the run.
        if not self._gi_freeze_whole_scan_prepass(scan):
            return 0
        frames = self._build_batch_frames(scan, pending)
        if not frames:
            return 0
        session, sink = self._get_streaming_session(scan, frames)
        if session is None:
            return 0
        count = 0
        for live in frames:
            self._wait_if_paused()        # pause between frames (streaming path)
            if self.command == 'stop':
                break
            # Per-frame status (replaces the debug-era 'Streaming N images...'):
            # 'Processing <name>' for per-file sources, 'Processing <master>
            # #<frame>' for multi-frame (Eiger) sources.  Filename middle-
            # truncated to <=30 chars; the #frame suffix stays visible.
            src = str(getattr(live, 'source_file', '') or '')
            name = self._middle_truncate(os.path.basename(src), max_len=30)
            if not name:
                name = f'frame {live.idx}'
            elif _raw_lives_in_source(src):
                name = f'{name} #{live.idx}'
            self.showLabel.emit(f'Processing {name}')
            sink.register(live)
            session.submit(frame_from_live_frame(live))
            count += 1
        return count

    def _get_streaming_session(self, scan, frames):
        """Build (once per scan) or return the persistent streaming session +
        its ``QtNexusSink``.  Returns ``(None, None)`` if the GI freeze scout
        fails (the batch is skipped with an advisory, as in the chunked path)."""
        if (self._streaming_session is not None
                and self._streaming_scan_id == id(scan)):
            return self._streaming_session, self._streaming_sink
        if self._streaming_session is not None:   # a different scan started
            self._close_reduction_session()
        standard_plan = self._plan_cache.get(scan, integrate_2d=not scan.skip_2d)
        n_workers = max(1, self.max_cores)
        executor = n_workers if n_workers > 1 else None
        cancel_token = self._cancel_token() if hasattr(self, "_cancel_token") else None
        # Batch brackets all frames (scout_union over first+last of the chunk);
        # live has no last frame at session open, so it freezes from the first
        # frame only (matches the legacy _process_one live path + the #75 advisory).
        _default_freeze = ("scout_union" if self.batch_mode else "first_frame") \
            if self.gi else None
        gi_freeze_mode = getattr(self, "gi_freeze_mode", _default_freeze)
        sink = QtNexusSink(self, scan, standard_plan, mask=self.mask)
        try:
            session = open_live_reduction_session(
                frames,
                standard_plan,
                scan_name=str(getattr(scan, "name", "scan")),
                global_mask=self.mask,
                integrator=scan._cached_integrator,
                poni=self.poni,
                executor=executor,
                cancel_token=cancel_token,
                gi_freeze_mode=gi_freeze_mode,
                sink=sink,
                execution="streaming",
            )
        except GIFreezeError as exc:
            self.showLabel.emit(
                'GI 2D scout frame is blank or the grid is degenerate: set '
                'Theta Motor to Manual and enter the incident angle, or check '
                'the mask / threshold.'
            )
            logger.warning('GI freeze scout failed for streaming batch: %s', exc)
            self._streaming_session = None
            self._streaming_sink = None
            self._streaming_scan_id = None
            return None, None
        self._streaming_session = session
        self._streaming_sink = sink
        self._streaming_scan_id = id(scan)
        return session, sink

    def _dispatch_batch_parallel(self, scan, pending):
        """Batch processing through the headless reduction executor.

        Phase 1 — Parallel integration:
            Build LiveFrame shells, then let ssrl_xrd_tools.run_reduction own
            the per-frame ThreadPoolExecutor work.

        Phase 2 — Serial HDF5 write:
            All completed frames are written to HDF5 under a single file_lock
            acquisition.  Skipped entirely in xye_only mode.
        """
        n_workers = min(self.max_cores, len(pending))
        if getattr(scan, "_cached_integrator", None) is None:
            # No cached integrator yet — fall back to serial dispatch
            # (the source integrator gets built on the first call).
            return self._dispatch_batch_serial(scan, pending)
        # F3: prewarm the bad-pixel mask on the main thread before
        # any worker reads it, so cache initialization isn't racy.
        # ``pending`` tuple shape: (img_file, img_number, img_data,
        # img_meta, bg_raw, t_read) — index 2 is the image.
        if (getattr(scan, '_cached_data_mask', None) is None
                and pending):
            self._prewarm_frame_mask(scan, pending[0][2])
        skip_2d = scan.skip_2d
        mask = self.mask
        gi = self.gi
        th_mtr = self.incidence_motor
        sample_orientation = self.sample_orientation
        tilt_angle = self.tilt_angle
        sync_live_scan_gi_settings(
            scan,
            incidence_motor=th_mtr,
            sample_orientation=sample_orientation,
            tilt_angle=tilt_angle,
        )
        standard_plan = self._plan_cache.get(scan, integrate_2d=not skip_2d)
        series_average = self.series_average

        frames = self._build_batch_frames(scan, pending)

        # ── Phase 1: headless parallel integration ───────────────────────────
        self.showLabel.emit(f'Integrating {len(frames)} images ({n_workers} workers)...')
        _t_phase1 = time.time()
        executor = n_workers if n_workers > 1 else None
        cancel_token = self._cancel_token() if hasattr(self, "_cancel_token") else None
        gi_freeze_mode = getattr(
            self,
            "gi_freeze_mode",
            "scout_union" if self.gi else None,
        )

        def _session_factory():
            return open_live_reduction_session(
                frames,
                standard_plan,
                scan_name=str(getattr(scan, "name", "scan")),
                global_mask=mask,
                integrator=scan._cached_integrator,
                poni=self.poni,
                executor=executor,
                cancel_token=cancel_token,
                chunk_size=len(frames) if frames else 1,
                gi_freeze_mode=gi_freeze_mode,
            )

        session_getter = getattr(self, "_get_reduction_session", None)
        close_session = not callable(session_getter)
        try:
            if close_session:
                session = _session_factory()
            else:
                session = session_getter(
                    self._reduction_session_key_for(scan, standard_plan, n_workers),
                    _session_factory,
                )
        except GIFreezeError as exc:
            # The GI freeze pre-pass scouts the grid when the session is built;
            # a blank/degenerate scout raises here.  Surface the fix and skip
            # this batch rather than aborting the run with an opaque traceback.
            self.showLabel.emit(
                'GI 2D scout frame is blank or the grid is degenerate: set '
                'Theta Motor to Manual and enter the incident angle, or check '
                'the mask / threshold.'
            )
            logger.warning('GI freeze scout failed for batch: %s', exc)
            return 0
        try:
            frames = reduce_live_frames(
                frames,
                standard_plan,
                scan_name=str(getattr(scan, "name", "scan")),
                global_mask=mask,
                integrator=scan._cached_integrator,
                poni=self.poni,
                session=session,
                cancel_token=cancel_token,
                chunk_size=len(frames) if frames else 1,
                gi_freeze_mode=gi_freeze_mode,
            )
        finally:
            if close_session:
                # Legacy chunked integration session (write is the separate
                # _save_to_nexus, not this session's sink) — preserve the
                # non-raising close; integration errors already re-raise.
                session.finish(raise_on_failure=False)
        # Precompute thumbnails in PARALLEL.  make_thumbnail is per-frame
        # numpy/scipy on the in-memory map_raw (the session doesn't clear it),
        # so it is thread-safe -- but it was the dominant *serial* cost left in
        # the batch path (~0.03s/frame, run on the main thread after the parallel
        # integration; ~17s of a 651-frame 2D batch, and the reason Int-1D batch
        # was 2x serial).  Fan it out over the same worker count as the
        # integration so batch wall-time matches/beats the old engine.
        #
        # Only compute thumbnails for frames that actually need a stored preview
        # (PERF-5): skip xye_only entirely (never persisted -- Phase 2 below is
        # gated off -- and never displayed: batch is silent), and skip 1D-only
        # (skip_2d) frames whose raw is reloadable from source (the Image Viewer
        # reloads the raw on demand via the per-frame source pointer).  The
        # writer (nexus_writer._write_per_frame_metadata) gates on the SAME
        # frame.can_skip_thumbnail() so a skipped frame is never lazily
        # re-thumbnailed at save time.  Keep thumbnails for 2D modes and for 1D
        # frames with no reloadable source (their only preview).
        _skip_2d = getattr(scan, 'skip_2d', False)
        thumb_frames = (
            [] if self.xye_only
            else [f for f in frames if not f.can_skip_thumbnail(_skip_2d)]
        )

        def _precompute_thumbnail(frame):
            frame.integrator = scan._cached_integrator
            try:
                frame.make_thumbnail(global_mask=mask)
            except Exception as e:
                logger.warning('Thumbnail precompute failed for image %s: %s',
                               frame.idx, e)

        if thumb_frames:
            if n_workers > 1 and len(thumb_frames) > 1:
                with ThreadPoolExecutor(max_workers=n_workers) as _thumb_pool:
                    list(_thumb_pool.map(_precompute_thumbnail, thumb_frames))
            else:
                for frame in thumb_frames:
                    _precompute_thumbnail(frame)
        with self._xye_lock:
            for frame in frames:
                self._xye_buffer.append((frame.idx, frame))
        _t_phase1 = time.time() - _t_phase1
        logger.info('[BATCH] Phase 1 (parallel integration): %d frames in %.2fs',
                    len(frames), _t_phase1)

        if not frames:
            return 0

        # ── Phase 2: serial HDF5 batch write ─────────────────────────────────
        if not self.xye_only:
            self.showLabel.emit(f'Writing {len(frames)} frames to HDF5...')
            _t_phase2 = time.time()
            _get_h5pool().pause(scan.data_file)
            try:
                with self.file_lock:
                    # Phase 2a: in-memory accumulation only (batch_save=True
                    # makes add_frame a pure in-memory op; no file I/O).
                    for frame in frames:
                        scan.add_frame(
                            frame=frame, calculate=False, update=True,
                            get_sd=True, set_mg=False, static=True, gi=gi,
                            th_mtr=th_mtr, series_average=series_average,
                            batch_save=True,
                        )
                    # Phase 2b: single batch flush — one slice-assign per
                    # stacked dataset for all frames in this batch.
                    # The writer owns its file handle now.
                    scan._save_to_nexus()
            finally:
                _get_h5pool().resume(scan.data_file)
            _t_phase2 = time.time() - _t_phase2
            logger.info('[BATCH] Phase 2 (HDF5 write): %d frames in %.2fs', len(frames), _t_phase2)

        # Flush buffered XYE writes for this batch.  P3: pass the
        # set of frame.idx values that actually landed in .nxs so
        # in-flight workers from a Stop'd batch don't leave orphan
        # XYE files for frames that were never published.
        _t_xye = time.time()
        published_idxs = {a.idx for a in frames}
        self._flush_xye_buffer(scan, published_idxs=published_idxs)
        _t_xye = time.time() - _t_xye
        if _t_xye > 0.01:
            logger.info('[BATCH] XYE flush: %d frames in %.2fs', len(frames), _t_xye)

        # PERF-3: each frame is now fully consumed for this batch — integrated
        # (Phase 1), thumbnail precomputed (Phase 1), written to .nxs
        # (Phase 2), XYE flushed.  scan.frames keeps the frame objects for the
        # rest of the scan, so without this their ~18 MB raw accumulates batch
        # over batch.  free_raw releases the raw only when it's losslessly
        # reloadable from source; a later viewer / reintegration read lazily
        # reloads (integrate_1d/2d, _lazy_load_raw).
        for frame in frames:
            frame.free_raw()

        return len(frames)

    def _process_one(self, scan, img_file, img_number, img_data, img_meta,
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
        # _apply_threshold_inline / _resolve_frame_mask docstrings for
        # why this is fast even with per-frame threshold filtering on.
        img_data = self._apply_threshold_inline(img_data)
        frame_mask = self._resolve_frame_mask(scan, img_data)
        frame = LiveFrame(
            img_number, img_data, poni=self.poni,
            scan_info=img_meta, static=True, gi=self.gi,
            th_mtr=self.incidence_motor, bg_raw=bg_raw,
            sample_orientation=self.sample_orientation,
            tilt_angle=self.tilt_angle,
            series_average=self.series_average,
            integrator=scan._cached_integrator,
            mask=frame_mask,
        )
        _t_frame = time.time() - _t1

        if self.gi:
            try:
                frame._get_incident_angle()
            except IncidenceAngleUnresolved as exc:
                # Refuse to integrate GI at a degenerate 0°.  Surface the
                # fix and skip the frame rather than emit a blank cake.
                self.showLabel.emit(
                    'GI needs an incidence angle: set Theta Motor to Manual '
                    'and enter the angle.'
                )
                logger.warning('Skipping GI frame %s: %s', img_file, exc)
                return

        _t2 = time.time()
        sync_live_scan_gi_settings(
            scan,
            incidence_motor=self.incidence_motor,
            sample_orientation=self.sample_orientation,
            tilt_angle=self.tilt_angle,
        )

        plan = self._plan_cache.get(scan, integrate_2d=not scan.skip_2d)
        cancel_token = self._cancel_token() if hasattr(self, "_cancel_token") else None

        def _session_factory():
            return open_live_reduction_session(
                [frame],
                plan,
                scan_name=str(getattr(scan, "name", "scan")),
                global_mask=self.mask,
                integrator=scan._cached_integrator,
                poni=self.poni,
                cancel_token=cancel_token,
                chunk_size=1,
                gi_freeze_mode="first_frame" if self.gi else None,
            )

        session_getter = getattr(self, "_get_reduction_session", None)
        close_session = not callable(session_getter)
        try:
            if close_session:
                session = _session_factory()
            else:
                session = session_getter(
                    self._reduction_session_key_for(scan, plan, 1),
                    _session_factory,
                )
        except GIFreezeError as exc:
            # Blank/degenerate GI scout (raised while building the session).
            # Mirror the IncidenceAngleUnresolved guidance and skip the frame.
            self.showLabel.emit(
                'GI 2D scout frame is blank or the grid is degenerate: set '
                'Theta Motor to Manual and enter the incident angle, or check '
                'the mask / threshold.'
            )
            logger.warning('GI freeze scout failed for frame: %s', exc)
            return
        try:
            reduce_live_frames(
                [frame],
                plan,
                scan_name=str(getattr(scan, "name", "scan")),
                global_mask=self.mask,
                integrator=scan._cached_integrator,
                poni=self.poni,
                session=session,
                cancel_token=cancel_token,
                chunk_size=1,
                gi_freeze_mode="first_frame" if self.gi else None,
            )
        finally:
            if close_session:
                # Legacy serial live integration session (write is the separate
                # _save_to_nexus) — preserve the non-raising close.
                session.finish(raise_on_failure=False)
        # Timing kept for parity with the legacy logging; the
        # standard path now does both 1D + 2D in one call so we
        # bundle the total under _t_1d.
        _t_1d = time.time() - _t2
        _t_2d = 0.0

        # ── GUI data (skip in batch mode — no one is looking) ────────────
        _t_h5_total = _t_h5_wait = _t_h5_write = 0.0
        if not self.batch_mode:
            self.data_1d[int(img_number)] = frame.copy_for_display(
                include_2d=False,
            )
            self.data_2d[int(img_number)] = {
                'map_raw': frame.map_raw,
                'bg_raw': frame.bg_raw,
                'mask': frame.mask,
                'int_2d': frame.int_2d,
                'gi_2d': frame.gi_2d,
                'thumbnail': None,
            }

        # ── In-memory accumulation (no disk I/O — batch flush handles that) ──
        if not self.xye_only:
            # Set source file as relative path from HDF5 dir for NeXus provenance
            if img_file:
                frame.source_file = os.path.abspath(str(img_file))
            else:
                frame.source_file = ""
            # source_frame_idx: per-source-file 0-based frame offset.
            # See the matching block in ``_dispatch_batch_parallel``
            # for the full rationale.  Eiger / HDF5 masters get
            # ``img_number - 1``; everything else stays at 0.
            if _raw_lives_in_source(img_file):
                frame.source_frame_idx = int(img_number) - 1
            else:
                frame.source_frame_idx = 0
            # For Eiger: raw frames already live in the master file — don't double-store them.
            frame.skip_map_raw = scan.skip_2d or _raw_lives_in_source(img_file)
            _t4 = time.time()
            # batch_save=True → pure in-memory (stash + index + bai_*).
            # The serial dispatcher calls scan._save_to_nexus once at
            # end-of-batch, so we don't pay per-frame write cost here.
            scan.add_frame(
                frame=frame, calculate=False, update=True,
                get_sd=True, set_mg=False, static=True, gi=self.gi,
                th_mtr=self.incidence_motor, series_average=self.series_average,
                batch_save=True,
            )
            _t_h5_total = time.time() - _t4
            _t_h5_wait = 0.0
            _t_h5_write = _t_h5_total

            # NOTE: no PERF-3 raw-free here.  This is the live/non-batch path,
            # where the display copy above (data_2d[idx]['map_raw']) keeps its
            # own ref to the raw for the whole scan — so freeing frame.map_raw
            # would NOT release memory (data_2d still pins it), it would only
            # cost an eager per-frame make_thumbnail (~7-15 ms each on the hot
            # loop) to keep the interval-flush thumbnail writer from reloading.
            # Net: pure regression on the lean 1D stream for zero RAM win.  The
            # raw-free lives only in the batch path (_dispatch_batch_parallel),
            # where data_2d is not populated so the free actually frees and the
            # thumbnail is already precomputed across workers.  Live-mode RAM is
            # bounded by the display-cache lifecycle, not by PERF-3.

        # ── XYE buffer (flushed at end of batch by the dispatcher) ──────
        _t5 = time.time()
        with self._xye_lock:
            self._xye_buffer.append((img_number, frame))
        _t_csv = time.time() - _t5

        _t_total = t_read + _t_frame + _t_1d + _t_2d + _t_h5_total + _t_csv
        # Merged per-frame line: timing + the user-facing "processed
        # <file>" annotation.  3-decimal precision so sub-10ms
        # components (in-memory add_frame, XYE buffer append) don't
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
            '[TIMING] %s: read=%.3fs frame_init=%.3fs '
            'int_1d=%.3fs int_2d=%.3fs add_frame=%.3fs csv=%.3fs '
            'total=%.3fs%s',
            _label, t_read, _t_frame, _t_1d, _t_2d,
            _t_h5_total, _t_csv, _t_total, _sub,
        )
        # In batch mode, suppress per-frame GUI signals — emit once at end
        # via process_scan's final ``sigUpdate.emit(-1)``.  Batch mode
        # exists precisely to skip GUI work during the run and let
        # the wrangler focus on integration throughput.
        if not self.batch_mode:
            # Publish the freshly-integrated frame so the main thread can
            # consume it without going back to disk.  See
            # static_scan_widget.update_data for the consumer side.
            self._published_frames[img_number] = frame
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
                                                meta_format=self.meta_ext, meta_dir=self.meta_dir)
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

        meta = read_image_metadata(self._eiger_master_path, meta_format=self.meta_ext, meta_dir=self.meta_dir) if self.meta_ext else {}

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
            meta = read_image_metadata(self.img_file, meta_format=self.meta_ext, meta_dir=self.meta_dir) if self.meta_ext else {}
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

            meta = read_image_metadata(fname, meta_format=self.meta_ext, meta_dir=self.meta_dir) if self.meta_ext else {}
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
        return read_image_metadata(img_file, meta_format=self.meta_ext, meta_dir=self.meta_dir)

    def subtract_bg(self, img_data, img_file, img_number, img_meta):
        bg = self.get_background(img_file, img_number, img_meta)
        try:
            img_data -= bg
        except ValueError:
            pass

    def initialize_scan(self):
        """If scan changes, initialize new LiveScan object.
        If mode is overwrite, replace existing HDF5 file, else append to it.
        """
        Path(self.h5_dir).mkdir(parents=True, exist_ok=True)
        fname = os.path.join(self.h5_dir, self.scan_name + '.nxs')
        # Eiger master files are pre-processed with the trailing
        # ``_master`` suffix stripped from scan_name (see
        # _get_next_eiger_frame). Without this sync, the wrangler
        # widget's self.fname (set from the original master filename
        # in image_wrangler.setup()) diverges from the actual scan
        # output path, and static_scan_widget.wrangler_finished
        # cannot find the generated file to reload at end of batch.
        self.fname = fname
        scan = LiveScan(self.scan_name,
                          data_file=fname,
                          static=True,
                          gi=self.gi,
                          incidence_motor=self.incidence_motor,
                          series_average=self.series_average,
                          single_img=self.single_img,
                          global_mask=self.mask,
                          # J2: share lock with wrangler save path
                          file_lock=self.file_lock,
                          **self.scan_args)
        scan.skip_2d = self.scan.skip_2d
        # N1: the project root -> entry/@source_base + relative raw source paths
        # in the writer (portable .nxs).  None -> absolute paths (back-compat).
        scan.source_base = getattr(self, "source_base", None)
        # v2 NeXus writer needs a DiffractometerGeometry to derive per-frame
        # rot1/rot2/rot3 and incidence-angle arrays from scan_data.  The
        # default is a two-circle convention using `tth` (detector arm) and
        # whatever `th_mtr` resolves to (sample tilt).  Override later from
        # the geometry UI panel when the user picks a non-default convention.
        scan.default_geometry()

        write_mode = self.write_mode
        if not os.path.exists(fname):
            write_mode = 'Overwrite'

        # Int 1D (XYE) writes ONLY .xye files (via the XYE flush); it must not
        # create or write the .nxs stack at all.  Skip all NeXus disk I/O here —
        # the per-batch ``scan._save_to_nexus`` is already gated off for
        # xye_only, so the scan object is used purely in-memory for integration.
        if not self.xye_only:
            _get_h5pool().pause(scan.data_file)
            try:
                with self.file_lock:
                    if write_mode == 'Append':
                        # v2 NeXus loader (the only one we support now).
                        scan.load_from_h5(replace=False, mode='a')
                        scan.skip_2d = self.scan.skip_2d
                        for (k, v) in self.scan_args.items():
                            setattr(scan, k, v)
                        existing_frames = scan.frames.index
                        if len(existing_frames) == 0:
                            scan.save_to_nexus(replace=True)
                    else:
                        scan.save_to_nexus(replace=True)
            finally:
                _get_h5pool().resume(scan.data_file)

        # Copy integration args (including GI modes) from the main scan.
        scan.bai_1d_args = self.scan.bai_1d_args.copy()
        scan.bai_2d_args = self.scan.bai_2d_args.copy()

        self.sigUpdateFile.emit(
            self.scan_name, fname,
            self.gi, self.incidence_motor, self.single_img,
            self.series_average
        )
        logger.info('***** New Scan *****')
        if self.xye_only:
            logger.info('Output (XYE) folder: %s',
                        os.path.join(self.h5_dir, self.scan_name))
        else:
            logger.info('Output file: %s', fname)

        return scan

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
                bg_meta = read_image_metadata(bg_file, meta_format=self.meta_ext, meta_dir=self.meta_dir)
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

                    bg_meta = read_image_metadata(bg_file, meta_format=self.meta_ext, meta_dir=self.meta_dir)
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
