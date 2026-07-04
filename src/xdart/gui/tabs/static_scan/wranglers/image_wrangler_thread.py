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
import numpy as np
from pathlib import Path
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# F1: shared boolean Filter grammar (unordered AND, '|'/OR, -term/NOT) —
# one compiled predicate replaces the old filter-encoded globs at all
# three sites (Image Directory, Eiger _master.h5 queue, BG Match).
from xrd_tools.core.filters import compile_filter as _compile_name_filter

_warned_bad_filters: set[str] = set()


def _name_filter(expr):
    """Compiled Filter predicate; a malformed expression warns once per
    expression and falls back to matching NOTHING.  Conservative on
    purpose: the old filter-encoded glob also matched nothing on garbage
    input, and a match-all fallback would process every file in the
    directory (or pick an arbitrary background at the BG Match site)."""
    try:
        return _compile_name_filter(expr)
    except ValueError as exc:
        key = str(expr)
        if key not in _warned_bad_filters:
            _warned_bad_filters.add(key)
            logger.warning("Invalid Filter expression %r (%s); matching "
                           "NO names until it is corrected", expr, exc)
        return lambda name: False

# pyFAI / fabio / h5py
import fabio
import h5py

# Qt imports
from pyqtgraph import Qt

# Project imports
from xdart.modules.live import LiveFrame, LiveScan, IncidenceAngleUnresolved
from xrd_tools.core import DEFAULT_MODE_KEY, FrameRecord
from xrd_tools.integrate.gid import gi_1d_output_axis_key
from xrd_tools.integrate.calibration import poni_to_integrator, get_detector
from xrd_tools.reduction import (
    FlushPolicy,
    GIFreezeError,
    GIMode,
    Integration1DPlan,
    ReductionPlan,
    prepare_gi_freeze,
)
from xrd_tools.session import FrameRecordStore
from xrd_tools.sources.image import ImageFileSource, TiffSeriesSource
from xrd_tools.io.image import read_image, count_frames
from xrd_tools.io.export import write_xye
from xrd_tools.io.nexus import find_nexus_image_dataset
from xrd_tools.io.metadata import read_image_metadata
from xdart.utils import get_series_avg
from xdart.utils.h5pool import get_pool as _get_h5pool
from xdart.modules.reduction import (
    freeze_live_scan_gi_ranges,
    frame_from_live_frame,
    open_live_reduction_session,
    open_live_scan_session,
    StandardPlanCache,
    reduce_live_frames,
    sync_live_scan_gi_settings,
)
from .qt_nexus_sink import QtNexusSink
from .wrangler_widget import wranglerThread

# Batch execution policy (PERF-4b/WS-X1).  Batch ALWAYS streams now: one
# persistent ReductionSession + QtNexusSink (submit-per-frame, single writer
# thread, thumbnail parallelized in the worker) — proven on the 651-frame Eiger
# scan at 8 cores to be byte-identical to and >= the old chunked path (2D 32.6s
# vs 38.8s, XYE 23.7s vs 25.4s, 1D ~equal).  4e retired the old read-chunk ->
# integrate -> Phase-2-write "chunked" dispatcher and its XDART_BATCH_EXECUTION
# escape hatch: there is one batch write path.

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

_LIVE_RECORD_STORE_MAX_ITEMS = 512


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _is_eiger_master(path):
    """Return True if path looks like an Eiger HDF5 master file (*_master.h5 / *_master.hdf5)."""
    return Path(path).stem.lower().endswith('_master')


def _paths_with_suffix(root, suffix, *, recursive=False):
    """Yield files below *root* whose name ends with *suffix*, ignoring case."""
    suffix = str(suffix or '').lower()
    if not suffix:
        return []
    root = Path(root)
    iterator = root.rglob('*') if recursive else root.glob('*')
    return (
        p for p in iterator
        if p.is_file() and p.name.lower().endswith(suffix)
    )


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

    Strips trailing _<digits> or -<digits> suffix from the stem to get scan_name.
    Falls back to (stem, None) when no numeric suffix is found.
    """
    return _split_scan_suffix(Path(fname).stem)


def _split_scan_suffix(stem):
    match = _FRAME_SUFFIX_PATTERN.match(stem)
    if not match:
        return stem, None
    return match.group(1), int(match.group(2))


def _gi_2d_range_keys(args):
    """Return the GI 2D range keys for the selected output mode."""
    mode = args.get('gi_mode_2d', 'qip_qoop')
    if mode == 'qip_qoop':
        return 'x_range', 'y_range'
    return 'radial_range', 'azimuth_range'


# The GI-scout cluster below (_padded_axis_range,
# _result_intensity_all_dummy, _freeze_gi_2d_ranges_from_result,
# _freeze_gi_1d_range_from_result, and the imageThread methods
# _scout_pending_frames, _build_scout, _freeze_gi_1d_auto_range,
# _freeze_gi_2d_auto_ranges) is LIVE PRODUCTION CODE: the streaming GI batch
# prepass calls _freeze_gi_1d_auto_range/_freeze_gi_2d_auto_ranges in
# _dispatch_batch_streaming to freeze the whole-scan common grid BEFORE
# dispatch (do NOT delete it as dead).  ssrl's ReductionSession has its own
# first-chunk freeze, which serves the serial/chunked paths; the streaming
# prepass deliberately freezes earlier (whole pending set, not first chunk).
# The GI live==batch==reload equivalence tests additionally bind these
# methods via MethodType to validate the ssrl freeze against this scout.
def _padded_axis_range(axis, pad_fraction=0.02):
    """Return a small coverage margin around an integrated axis' finite extent.

    The 2% margin is load-bearing: a fresh per-frame integration can land a few
    bins beyond a scout's auto-range extent (binning discretization), and
    without it the frozen range would CLIP that real data.  The empty bins the
    margin creates beyond the real data should be NaN-filled (not a spurious
    dummy) so they are not plotted — see the NaN-empty follow-up.

    Returns ``None`` when the axis is missing, has no finite samples, or is
    *collapsed* (span <= 0 — every finite value identical).  A collapsed
    axis means the scout integration was degenerate (e.g. GI at a 0°
    incidence); freezing a tiny range from it would clamp every subsequent
    frame onto that collapsed grid and blank the whole scan.  Returning
    ``None`` leaves the range unfrozen so the caller can surface the problem
    instead of silently squashing the output.
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
    (:func:`xrd_tools.integrate.gid.gi_1d_output_axis_key`): ``azimuth_range``
    for q_oop/exit_angle (out-of-plane output), else ``radial_range``.  Freezing
    the wrong key leaves the output axis auto-ranging per incidence → a
    non-uniform stack the writer rejects."""
    return gi_1d_output_axis_key(gi_mode_1d)


def _freeze_gi_1d_range_from_result(args, result, gi_mode_1d=None):
    """Freeze the missing GI 1D *output-axis* range from one scout result so all
    frames share one axis.  Picks radial_range vs azimuth_range by mode (see
    :func:`_gi_1d_output_range_key`)."""
    mode = gi_mode_1d if gi_mode_1d is not None else args.get('gi_mode_1d')
    key = _gi_1d_output_range_key(mode)
    if args.get(key) is not None:
        return False
    rng = _padded_axis_range(getattr(result, 'radial', None))
    if rng is None:
        return False
    if mode in ('exit_angle', 'chi_gi'):
        rng = (max(float(rng[0]), -180.0), min(float(rng[1]), 180.0))
    args[key] = rng
    return True


# ---------------------------------------------------------------------------
# Natural sort helpers
# ---------------------------------------------------------------------------

# Pre-compiled regex patterns (avoids recompilation on every sort key call)
_INT_PATTERN = re.compile(r'(\d+)')
_FLOAT_PATTERN = re.compile(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)')
_FRAME_SUFFIX_PATTERN = re.compile(r'^(.*?)[_-](\d+)$')

# How many integrated frames to accumulate between v2 _save_to_nexus
# Live-save cadence and threshold-sentinel constants moved to the
# wranglerThread base class (wrangler_widget.py) in May 2026 — refer
# via ``self.LIVE_SAVE_INTERVAL`` / the base's module-level
# ``_THRESHOLD_NAN`` (re-imported here for back-compat with old SPEC
# wrangler scans that may pickle/unpickle constants by name).

# Number of frames the background prefetch worker may read ahead of the main
# collect loop (the prefetch queue's maxsize).  This is the read‖reduce OVERLAP
# budget: the collect loop blocks on dispatch (reduce+write backpressure) for
# seconds per chunk, and the prefetcher can only run ahead this many frames
# before its queue fills and it stalls — so a tiny value makes collect_read and
# dispatch ADDITIVE instead of overlapped ([PERF-SUMMARY], 2026-06-15: Eiger 1D
# read is decompression-bound ~24 ms/frame, single-thread prefetch floor ~16 s,
# but additive gives ~25 s).  Trade-off: 18 MB/frame, so a large queue costs RAM
# and can contend with the writer.  Env-tunable for perf experiments; default 4
# preserves the prior behaviour.  (The bulk-read path is a dead end for per-frame
# -compressed Eiger data — it does NOT amortize decompression.)
try:
    _PREFETCH_QUEUE_SIZE = max(1, int(os.environ.get("XDART_PREFETCH_QUEUE_SIZE", "4")))
except (TypeError, ValueError):
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
        # GENERIC-DETECTOR FIX: when a Run adopts a loaded processed scan's
        # geometry (image_wrangler._adopt_loaded_scan_run_inputs), the wrangler
        # also hands over the restored PIXEL-BEARING integrator keyed on the
        # adopted poni.  The poni-identity rebuild block below REUSES it instead
        # of rebuilding a pixel-less integrator from the (name-only) PONI.  Both
        # default None (a normal poni-file Run rebuilds as before).
        self._adopted_poni = None
        self._adopted_integrator = None
        self._adopted_fiber_integrator = None
        self.inp_type = inp_type
        self.img_file = img_file
        self.img_dir = img_dir
        self.include_subdir = include_subdir
        self.img_ext = img_ext
        self.series_average = series_average
        self.meta_ext = meta_ext
        # Optional explicit SPEC search dir.  Set by the wrangler
        # widget (set_meta_dir / setup) — None / '' falls back to
        # the xrd_tools default heuristic.  Threaded through
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

        self.user = None
        self.mask = None
        self.detector_shape = None   # full-res detector (raw) shape (H, W)
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
        self._eiger_metadata_cache = {}  # master metadata is stable across frames
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
        self._append_skip_frames_by_scan = {}
        self._append_skip_without_reading = 0
        self._discovered_frame_count = 0
        self._skip_reason_counts = Counter()

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
        self._eiger_metadata_cache.clear()
        # Start each run with an empty producer->consumer display slot so a frame
        # the GUI never popped (a missed/raced consume, or a frame published after
        # the final flush) can't leak a whole LiveFrame (~18 MB raw) into the next
        # run.  Safe to clear here: the prior run's consumer is done and this run
        # hasn't published yet.
        self._published_frames.clear()
        self._prefetch_stop_prior()     # tear down any lingering prefetcher
        self._prefetch_queue = None
        self._prefetch_thread = None
        self._prefetch_stop_evt = None
        self._prefetch_error = None
        self._append_skip_frames_by_scan = {}
        self._append_skip_without_reading = 0
        self._discovered_frame_count = 0
        self._skip_reason_counts = Counter()
        # Per-run perf accumulators -> [PERF-SUMMARY] at end of run.  Breaks the
        # worker time into consumer read(queue-wait)/integrate/write + the actual
        # background prefetch I/O, so ONE run shows the bottleneck (read vs
        # integrate vs write) without bisecting.  Written by _process_one (main
        # thread) and _prefetch_worker (bg thread) on disjoint keys.
        self._perf = {
            # collect/flush level (runs for BOTH the streaming and serial paths):
            'collect_read': 0.0,      # main-thread queue-wait on the prefetcher
            'dispatch': 0.0,          # _dispatch_batch: reduce + write for the chunk
            'dispatch_frames': 0,
            'prefetch_io': 0.0,       # actual background HDF5 block I/O
            'prefetch_frames': 0,
            # serial-path per-frame breakdown (only populated by _process_one):
            'frame': 0.0, '1d': 0.0, '2d': 0.0, 'h5': 0.0, 'csv': 0.0, 'n': 0,
        }
        self.detector = get_detector(self.poni.detector) if self.poni.detector else None
        self.sub_label = ''
        det_mask = self.detector.mask if self.detector is not None else None  # pyFAI .mask property
        if self.mask_file and os.path.exists(self.mask_file):
            try:
                custom_mask = np.asarray(read_image(self.mask_file), dtype=bool)
                # Validate the Mask File against the detector FRAME shape: the
                # built-in mask shape, or (for detectors with no built-in mask,
                # e.g. RayonixMx225) the geometry shape.  A shape mismatch can't
                # index the frame, so warn + ignore -- now consistent for BOTH
                # cases (previously only checked when a built-in mask existed),
                # which also keeps the persisted detector_shape trustworthy
                # (adversarial review: a mismatched mask must not be stored as
                # the detector shape).
                if det_mask is not None:
                    _ref_shape = tuple(det_mask.shape)
                elif self.detector is not None:
                    _gs = (getattr(self.detector, "shape", None)
                           or getattr(self.detector, "max_shape", None))
                    _ref_shape = tuple(_gs) if _gs is not None else None
                else:
                    _ref_shape = None
                if _ref_shape is not None and tuple(custom_mask.shape) != _ref_shape:
                    logger.warning('Mask file shape %s does not match detector shape %s — ignoring custom mask',
                                  custom_mask.shape, _ref_shape)
                else:
                    det_mask = det_mask | custom_mask if det_mask is not None else custom_mask
            except Exception as e:
                logger.warning('Could not load mask file %s: %s', self.mask_file, e)
        self.mask = np.flatnonzero(det_mask) if det_mask is not None else None
        # The full-res raw frame shape (H, W) the flat mask indices index into —
        # carried onto the scan + persisted so a reloaded thumbnail-only scan can
        # map the detector gap mask into thumbnail coordinates.  Prefer det_mask's
        # shape (a mismatched Mask File was rejected above, so it's the true frame
        # shape); otherwise capture the detector geometry shape even with NO mask
        # (future per-frame masks / cold reloads), per review.
        if det_mask is not None:
            self.detector_shape = tuple(det_mask.shape)
        elif self.detector is not None:
            _gs = (getattr(self.detector, "shape", None)
                   or getattr(self.detector, "max_shape", None))
            self.detector_shape = tuple(_gs) if _gs is not None else None
        else:
            self.detector_shape = None
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
        _perf = getattr(self, '_perf', None)
        if _perf and (_perf['dispatch_frames'] or _perf['prefetch_frames'] or _perf['n']):
            _df = max(_perf['dispatch_frames'], 1)
            _pf = max(_perf['prefetch_frames'], 1)
            # collect_read = main-thread queue-wait on the prefetcher; dispatch =
            # reduce+write per chunk; prefetch_io = real background HDF5 block I/O.
            # If collect_read ~= prefetch_io the bottleneck is raw read; if dispatch
            # dominates it's reduce/write.
            logger.info(
                '[PERF-SUMMARY] dispatch_frames=%d | collect_read(queue-wait)=%.2fs '
                'dispatch(reduce+write)=%.2fs | prefetch_io=%.2fs (%d frames, %.1f ms/frame)'
                ' | per-frame ms: read=%.1f dispatch=%.1f',
                _perf['dispatch_frames'], _perf['collect_read'], _perf['dispatch'],
                _perf['prefetch_io'], _perf['prefetch_frames'],
                _perf['prefetch_io'] / _pf * 1e3,
                _perf['collect_read'] / _df * 1e3, _perf['dispatch'] / _df * 1e3,
            )
            if _perf['n']:   # serial path only: per-frame integrate/write split
                _n = _perf['n']
                logger.info(
                    '[PERF-SUMMARY serial] %d frames: frame_init=%.2fs int_1d=%.2fs '
                    'int_2d=%.2fs h5_write=%.2fs csv=%.2fs',
                    _n, _perf['frame'], _perf['1d'], _perf['2d'], _perf['h5'],
                    _perf['csv'],
                )
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

    def _install_run_integrator(self, scan):
        """Build (or reuse) the scan's cached AzimuthalIntegrator for this run.

        Called when ``self.poni`` differs from the scan's currently-cached poni
        (a fresh scan, or a user-loaded calibration).  The default path rebuilds
        from ``self.poni`` via ``poni_to_integrator``.

        GENERIC-DETECTOR FIX: a PONI dataclass carries only a detector *name*, so
        rebuilding for an unnamed/generic detector yields a pyFAI integrator with
        ``_pixel1``/``_pixel2`` = None — and the reduction then crashes in
        ``calc_cartesian_positions`` (``NoneType * float``).  When this run ADOPTED
        a loaded processed scan's geometry (image_wrangler._adopt_loaded_scan_run_inputs),
        the wrangler also handed over that scan's restored PIXEL-BEARING integrator,
        keyed on the adopted poni object.  Reuse it here whenever ``self.poni`` IS
        that adopted poni (identity check) so the pixel size survives.  A
        genuinely-new user-loaded .poni is a different object, so it still rebuilds.
        """
        adopted_ai = getattr(self, "_adopted_integrator", None)
        if (adopted_ai is not None
                and self.poni is getattr(self, "_adopted_poni", None)):
            scan._cached_integrator = adopted_ai
            scan._cached_fiber_integrator = getattr(
                self, "_adopted_fiber_integrator", None)
            logger.info(
                "[RUN-CAL] reusing restored pixel-bearing integrator for %s "
                "(generic detector — poni rebuild would drop pixel size)",
                getattr(scan, "name", "scan"))
        else:
            scan._cached_integrator = poni_to_integrator(self.poni)
            scan._cached_fiber_integrator = None
        scan._cached_poni = self.poni

    def _append_skip_enabled(self):
        return (getattr(self, "write_mode", None) == 'Append'
                and not getattr(self, "xye_only", False))

    def _append_output_number(self, img_number):
        if getattr(self, "series_average", False):
            return 1
        return 1 if img_number is None else img_number

    def _record_discovered_frame(self, count=1):
        self._discovered_frame_count = (
            getattr(self, "_discovered_frame_count", 0) + count)

    def _record_skip_reason(self, reason, count=1):
        reasons = getattr(self, "_skip_reason_counts", None)
        if reasons is None:
            reasons = Counter()
            self._skip_reason_counts = reasons
        reasons[str(reason)] += count

    @contextmanager
    def _optional_lock(self, lock):
        if lock is None:
            yield
        else:
            with lock:
                yield

    def _scan_frame_index_snapshot(self, scan):
        frames = getattr(scan, "frames", None)
        lock = getattr(scan, "scan_lock", None)
        with self._optional_lock(lock):
            index = getattr(frames, "index", ())
            return set(index if index is not None else ())

    def _warn_append_snapshot_failed(self, scan_name, out_path, exc):
        warned = getattr(self, "_append_skip_snapshot_warnings", None)
        if warned is None:
            warned = set()
            self._append_skip_snapshot_warnings = warned
        key = str(scan_name)
        if key in warned:
            return
        warned.add(key)
        logger.warning(
            "append skip snapshot unavailable for %s; proceeding without "
            "pre-read skips: %s",
            out_path, exc,
        )

    def _remember_append_skip_snapshot(self, scan_name, frame_index=None, *, scan=None):
        if not self._append_skip_enabled() or scan_name is None:
            return
        cache = getattr(self, "_append_skip_frames_by_scan", None)
        if cache is None:
            cache = {}
            self._append_skip_frames_by_scan = cache
        try:
            existing = (self._scan_frame_index_snapshot(scan)
                        if scan is not None
                        else set(frame_index or ()))
        except Exception as exc:
            out_path = self._append_output_path(scan_name)
            self._warn_append_snapshot_failed(scan_name, out_path, exc)
            existing = set()
        cache[str(scan_name)] = existing

    def _append_output_path(self, scan_name):
        return os.path.join(getattr(self, "h5_dir", ""), str(scan_name) + '.nxs')

    def _append_run_start_scan_names(self):
        if not self._append_skip_enabled():
            return []

        names = []
        seen = set()

        def add(name):
            if name is None:
                return
            key = str(name)
            if key and key not in seen:
                seen.add(key)
                names.append(key)

        img_file = getattr(self, "img_file", None)
        inp_type = getattr(self, "inp_type", None)
        img_ext = (getattr(self, "img_ext", "") or "").lower().lstrip(".")

        if inp_type == 'Image Directory':
            img_dir = getattr(self, "img_dir", None)
            if not img_dir:
                return names
            match = _name_filter(getattr(self, "file_filter", ""))
            root = Path(img_dir)
            include_subdir = bool(getattr(self, "include_subdir", False))
            if img_ext in ('h5', 'hdf5'):
                suffix = f'_master.{img_ext}'
                candidates = _paths_with_suffix(
                    root, suffix, recursive=include_subdir)
                for path in natural_sort_ints([str(p) for p in candidates]):
                    p = Path(path)
                    if match(p.name[:-len(suffix)]):
                        add(self._eiger_scan_name(p))
            elif img_ext and img_ext not in ('nxs',):
                suffix = f'.{img_ext}'
                candidates = _paths_with_suffix(
                    root, suffix, recursive=include_subdir)
                for path in natural_sort_ints([str(p) for p in candidates]):
                    p = Path(path)
                    if match(p.name[:-len(suffix)]):
                        add(_get_scan_info(p)[0])
            return names

        if img_file:
            ext = Path(img_file).suffix.lower().lstrip(".")
            if _is_eiger_master(img_file) or ext in ('h5', 'hdf5', 'nxs'):
                add(self._eiger_scan_name(img_file))
            else:
                add(_get_scan_info(img_file)[0])
        if not names:
            add(getattr(self, "scan_name", None))
        return names

    def _new_append_scan_for_snapshot(self, scan_name, out_path):
        return LiveScan(
            scan_name,
            data_file=out_path,
            static=True,
            gi=getattr(self, "gi", False),
            incidence_motor=getattr(self, "incidence_motor", None),
            series_average=getattr(self, "series_average", False),
            single_img=getattr(self, "single_img", False),
            global_mask=getattr(self, "mask", None),
            detector_shape=getattr(self, "detector_shape", None),
            file_lock=getattr(self, "file_lock", None),
            **(getattr(self, "scan_args", {}) or {}),
        )

    def _prime_append_skip_snapshots_for_run(self):
        if not self._append_skip_enabled():
            return
        cache = getattr(self, "_append_skip_frames_by_scan", None)
        if cache is None:
            cache = {}
            self._append_skip_frames_by_scan = cache

        for scan_name in self._append_run_start_scan_names():
            if scan_name in cache:
                continue
            out_path = self._append_output_path(scan_name)
            # The finish handler uses thread.fname even when every frame was
            # skipped before initialize_scan() could run.
            self.fname = out_path
            if not os.path.exists(out_path):
                cache[scan_name] = set()
                continue
            try:
                scan = self._new_append_scan_for_snapshot(scan_name, out_path)
                scan.load_from_h5(replace=False, mode='r')
                self._remember_append_skip_snapshot(scan_name, scan=scan)
            except Exception as exc:
                self._warn_append_snapshot_failed(scan_name, out_path, exc)
                cache[scan_name] = set()

    def _series_average_append_blocker(self):
        """MEM-1c: refuse a silently-empty series-average Append run.

        ``_append_output_number`` collapses EVERY source frame of a series
        average to output frame 1.  In Append mode, if that averaged output
        already exists on disk, ``_should_skip_before_read`` would skip every
        source frame and the run would produce NOTHING — a silent no-op that
        only logs a benign INFO line and looks like success.  Detect it up
        front (the append-skip snapshots are already primed) and return an
        actionable reason so the run is refused loudly instead.  Returns the
        user-facing message, or ``None`` when the run may proceed.
        """
        if not (getattr(self, "series_average", False)
                and self._append_skip_enabled()):
            return None
        collapsed = self._append_output_number(None)   # series-average => 1
        for scan_name in self._append_run_start_scan_names():
            if collapsed in self._append_skip_snapshot(scan_name):
                return (
                    f"Averaged output already exists for '{scan_name}' — the "
                    "whole series would be skipped and nothing written.  "
                    "Switch write mode to Replace, or clear the target, before "
                    "running a series average.")
        return None

    def _append_skip_snapshot(self, scan_name):
        """Return this run's append-skip frame snapshot for *scan_name*.

        Run start primes this cache from disk in read-only mode.  The per-frame
        path is intentionally a pure in-memory lookup: no HDF5 open, no writer
        mode, and no exception that can kill the run.
        """
        if not self._append_skip_enabled() or scan_name is None:
            return set()
        key = str(scan_name)
        cache = getattr(self, "_append_skip_frames_by_scan", None)
        if cache is None:
            cache = {}
            self._append_skip_frames_by_scan = cache
        if key in cache:
            return cache[key]
        cache[key] = set()
        return cache[key]

    def _should_skip_before_read(self, scan_name, img_number):
        output_img_number = self._append_output_number(img_number)
        if output_img_number in self._append_skip_snapshot(scan_name):
            self._append_skip_without_reading = (
                getattr(self, "_append_skip_without_reading", 0) + 1)
            self._record_skip_reason("already processed")
            return True
        return False

    def _format_skip_reasons(self):
        reasons = getattr(self, "_skip_reason_counts", Counter())
        if not reasons:
            return ""
        parts = []
        for reason, count in reasons.most_common():
            parts.append(f"{reason} ({count})" if count != 1 else reason)
        return "; ".join(parts)

    def _report_run_skip_summary(self, files_processed):
        skipped = getattr(self, "_append_skip_without_reading", 0)
        if skipped:
            logger.info(
                "append: skipping %d already-processed frame(s) without reading",
                skipped,
            )

        discovered = getattr(self, "_discovered_frame_count", 0)
        if (discovered <= 0 or files_processed >= discovered
                or getattr(self, "command", None) == 'stop'):
            return
        if getattr(self, "series_average", False) and files_processed > 0:
            return
        msg = (
            f"{files_processed} of {discovered} discovered frame(s) processed"
        )
        reasons = self._format_skip_reasons()
        if reasons:
            msg = f"{msg}: {reasons}"
        only_already_processed = (
            files_processed == 0
            and discovered > 0
            and getattr(self, "_skip_reason_counts", Counter())
            == Counter({"already processed": discovered})
        )
        log = logger.info if only_already_processed else logger.warning
        log(msg)
        try:
            self.showLabel.emit(msg)
        except Exception:
            logger.debug("showLabel emit failed for zero-frame warning",
                         exc_info=True)

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
        # XDART_LIVE_EXECUTION flag is observable).  Batch always streams now.
        # DEBUG: developer diagnostics, not run output.
        logger.debug('Execution policy: batch_mode=%s  batch=streaming  live=%s',
                     self.batch_mode, self._live_execution())
        self._prime_append_skip_snapshots_for_run()

        # MEM-1c: refuse a series-average Append run whose averaged output
        # already exists (would silently skip everything and write nothing).
        # Only series-average runs can collapse this way, so gate the check on
        # it (also keeps non-series-average paths free of the lookup).
        _blocker = (self._series_average_append_blocker()
                    if getattr(self, "series_average", False) else None)
        if _blocker:
            logger.error("run refused: %s", _blocker)
            _emit = getattr(getattr(self, "showLabel", None), "emit", None)
            if callable(_emit):
                try:
                    _emit(_blocker)
                except Exception:
                    logger.debug("showLabel emit failed for run blocker",
                                 exc_info=True)
            self.command = 'stop'
            return

        # ── Phase 1 & 2: collect then process all existing images ─────────────
        pending = []  # [(img_file, img_number, img_data, img_meta, bg_raw)]
        pending_avg_count = 0
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
                if not self.batch_mode and pending:
                    logger.info('Collected %d image(s) in %.2fs',
                                len(pending), _t_read_accum)
                break  # initial glob exhausted — move on to processing
            if img_file is not None:
                fname = os.path.splitext(os.path.basename(img_file))[0]
                # When the input is a multi-frame container (HDF5/NeXus/Eiger master),
                # include the frame index so progress is visible.
                _ext = Path(img_file).suffix.lower()
                _multi = _ext in ('.h5', '.hdf5', '.nxs') or _is_eiger_master(img_file)
                _label = (f'Collecting {self._middle_truncate(fname)} [frame {img_number}]'
                          if _multi else
                          f'Collecting {self._middle_truncate(fname)}')
                if self.batch_mode:
                    # Batch: the label is the only progress feedback while the
                    # whole pending set is read up front.
                    self.showLabel.emit(_label)
                else:
                    # Live: frames dispatch as they arrive, so the completion
                    # status (sink) supersedes this within ms -- showing it
                    # just flickered.  Keep it greppable at DEBUG; a single
                    # summary INFO is logged when collection finishes.
                    logger.debug(_label)
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
                    pending_avg_count = 0
                # Catch the case where pending was already drained by
                # the per-iteration dispatch cadence below: the old
                # scan may have integrated-but-unsaved frames in
                # memory whose save_to_nexus call never fired.
                if (scan is not None
                        and not self.xye_only
                        and self._frames_since_save > 0):
                    _n_swap = self._frames_since_save     # reset by the tail
                    if self.flush_serial_tail(scan, force=True):
                        logger.info(
                            '[SAVE-ON-SWAP] %d frames flushed for %s',
                            _n_swap, scan.name,
                        )
                scan = self.initialize_scan()
                self._active_scan = scan      # Pause: serial-flush handle
                _cached_poni = None

            # Rebuild cached AzimuthalIntegrator when poni identity changes
            if self.poni is not _cached_poni:
                self._install_run_integrator(scan)
                _cached_poni = self.poni
                self._cached_gi_incident_angle = None

            series_average = bool(getattr(self, "series_average", False))
            output_img_number = 1 if series_average else img_number
            if output_img_number in scan.frames.index:
                self._record_skip_reason("already processed at dispatch")
                if self.single_img and not is_eiger:
                    self.sigUpdate.emit(img_number)
                    break
                continue

            bg_raw = self.get_background(img_file, img_number, img_meta)
            # Stash the per-frame read time on the tuple so the per-frame
            # TIMING log can show it (otherwise it gets lumped into the
            # batch [FLUSH] line and hides per-frame variance).
            entry = (img_file, img_number, img_data, img_meta,
                     bg_raw, _t_read_this)
            if series_average:
                pending_avg_count = imageThread._append_series_average_pending(
                    self, pending, entry, pending_avg_count)
            else:
                pending.append(entry)

            if self.single_img and not is_eiger:
                break

            # ── Submit cadence ──────────────────────────────────────────────
            # Streaming batch owns its own in-flight bound and save cadence, so
            # the old "read 64/256 frames before dispatch" buffer only destroyed
            # read||reduce overlap.  Submit every frame as soon as it is read;
            # QtNexusSink/FlushPolicy still batch persistence, and batch remains
            # display-silent until the final sigUpdate(-1).
            flush_size = 1
            if pending and len(pending) >= flush_size and not series_average:
                if self.batch_mode:
                    self.showLabel.emit(
                        f'Integrating {len(pending)} frame(s)...'
                    )
                _t_disp = time.time()
                dispatched = self._dispatch_batch(scan, pending)
                files_processed += dispatched
                _disp_dt = time.time() - _t_disp
                _perf = getattr(self, '_perf', None)
                if _perf is not None:
                    _perf['collect_read'] += _t_read_accum
                    _perf['dispatch'] += _disp_dt
                    _perf['dispatch_frames'] += dispatched
                if _t_read_accum >= 0.5 or _disp_dt >= 0.5:
                    logger.info(
                        '[DISPATCH] %d frames  read=%.2fs  dispatch=%.2fs',
                        len(pending), _t_read_accum, _disp_dt,
                    )
                pending = []
                pending_avg_count = 0
                _t_read_accum = 0.0

        # Process whatever is left.  force_save=True so any remaining
        # _frames_since_save tail in live mode is flushed to disk.
        if pending and scan is not None and self.command != 'stop':
            _t_disp = time.time()
            dispatched = self._dispatch_batch(
                scan, pending, force_save=True,
            )
            files_processed += dispatched
            pending_avg_count = 0
            _disp_dt = time.time() - _t_disp
            _perf = getattr(self, '_perf', None)
            if _perf is not None:
                _perf['collect_read'] += _t_read_accum
                _perf['dispatch'] += _disp_dt
                _perf['dispatch_frames'] += dispatched
            logger.info(
                '[FLUSH-FINAL] %d frames  read=%.2fs  dispatch=%.2fs',
                len(pending), _t_read_accum, _disp_dt,
            )
        elif (scan is not None and not self.xye_only
              and self._frames_since_save > 0 and self.command != 'stop'):
            # Pending was empty but the live-save batcher has unflushed frames
            # (last batch hit the divisor exactly).  Force a save before leaving
            # the collect loop.
            self.flush_serial_tail(scan, force=True)

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
                    self._install_run_integrator(scan)
                    _cached_poni = self.poni
                    self._cached_gi_incident_angle = None

                if img_number in scan.frames.index:
                    self._record_skip_reason("already processed at dispatch")
                    continue

                bg_raw = self.get_background(img_file, img_number, img_meta)
                # Process immediately — single-threaded for low latency.
                # _process_one uses add_frame(batch_save=True), so the
                # disk save still rides on _frames_since_save below.
                self._process_one(scan, img_file, img_number, img_data, img_meta, bg_raw)
                files_processed += 1
                self._frames_since_save += 1
                if self.xye_only:
                    # No .nxs save in this mode -- drain the XYE buffer per
                    # frame so output appears as the watch processes files.
                    self._flush_xye_buffer(scan)
                self.flush_serial_tail(scan)

        # Final flush on exit (live-watch tail or stop request) so the
        # last few frames aren't lost.
        if scan is not None and self.xye_only:
            self._flush_xye_buffer(scan)
        self.flush_serial_tail(scan, force=True)

        # In batch mode, emit a single final signal so the GUI can refresh
        self.files_processed = files_processed
        self._last_files_processed = files_processed
        if self.batch_mode and files_processed > 0:
            self.sigUpdate.emit(-1)
        report_skip_summary = getattr(self, "_report_run_skip_summary", None)
        if callable(report_skip_summary):
            report_skip_summary(files_processed)
        logger.info('Total Files Processed: %d', files_processed)

    # ── Batch dispatch ────────────────────────────────────────────────────────

    @staticmethod
    def _average_numeric_metadata(current, incoming, old_count, new_count):
        out = dict(current or {})
        for key in list(out):
            try:
                out[key] = (
                    float(out[key]) * old_count
                    + float((incoming or {}).get(key, 0.0))
                ) / new_count
            except (TypeError, ValueError):
                pass
        return out

    @staticmethod
    def _average_payload(current, incoming, old_count, new_count):
        if current is None:
            return None
        if incoming is None:
            return current
        try:
            return (
                np.asarray(current, dtype=float) * old_count
                + np.asarray(incoming, dtype=float)
            ) / new_count
        except (TypeError, ValueError):
            return current

    @staticmethod
    def _copy_average_payload(value):
        if value is None:
            return None
        return np.asarray(value, dtype=float).copy()

    def _append_series_average_pending(self, pending, entry, count):
        """Fold one source frame into the pending Average Scan mean.

        ``pending`` keeps the normal dispatcher tuple shape, but holds exactly
        one running-mean entry.  This mirrors the old queue feeder without
        retaining every detector image until the streaming session opens.
        """

        img_file, _img_number, img_data, img_meta, bg_raw, t_read = entry
        if count <= 0 or not pending:
            pending[:] = [(
                img_file,
                1,
                imageThread._copy_average_payload(img_data),
                dict(img_meta or {}),
                imageThread._copy_average_payload(bg_raw),
                float(t_read or 0.0),
            )]
            return 1

        old_file, _old_number, old_data, old_meta, old_bg, old_t = pending[0]
        new_count = count + 1
        pending[0] = (
            img_file or old_file,
            1,
            imageThread._average_payload(old_data, img_data, count, new_count),
            imageThread._average_numeric_metadata(
                old_meta, img_meta, count, new_count),
            imageThread._average_payload(old_bg, bg_raw, count, new_count),
            float(old_t or 0.0) + float(t_read or 0.0),
        )
        return new_count

    def _series_average_pending(self, pending):
        if not bool(getattr(self, "series_average", False)) or len(pending) <= 1:
            return list(pending)
        averaged = []
        count = 0
        for entry in pending:
            count = imageThread._append_series_average_pending(
                self, averaged, entry, count)
        return averaged

    def _dispatch_batch(self, scan, pending, *, force_save=False):
        """Process a list of pending images — parallel in batch mode, serial otherwise.

        ``force_save`` only affects the serial path (live mode); the
        parallel path always saves at end of batch by construction
        (the whole point of batch mode is one big save per dispatch).
        """
        self._maybe_warn_live_gi_clip()
        if self.batch_mode:
            # 4e: batch is one write path — always the streaming session.
            return self._dispatch_batch_streaming(scan, pending)
        # Live (non-batch): #3 routes it through the SAME streaming session +
        # QtNexusSink (which does the per-frame display publish for live), behind
        # the live flag; the proven per-frame _process_one path is the default.
        if self._live_execution() == "streaming":
            return self._dispatch_batch_streaming(scan, pending)
        return self._dispatch_batch_serial(scan, pending, force_save=force_save)

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

    # GI-scout cluster: LIVE PRODUCTION CODE (see the authoritative cluster
    # note near _padded_axis_range).  _freeze_gi_1d_auto_range /
    # _freeze_gi_2d_auto_ranges run in the streaming batch prepass
    # (_dispatch_batch_streaming); _scout_pending_frames / _build_scout are
    # their selection/build helpers, also exercised directly by the
    # GI-equivalence tests.
    def _scout_pending_frames(self, pending):
        """Return bounded representative pending entries for the GI freeze.

        Selection helper for :meth:`_freeze_gi_1d_auto_range` /
        :meth:`_freeze_gi_2d_auto_ranges` (the streaming batch prepass —
        live production code; see the cluster note near
        ``_padded_axis_range``).  ssrl's ReductionSession has its own
        first-chunk freeze for the serial/chunked paths.  Real-data tests
        also inspect this selection directly.
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
        if not (self.gi and self.batch_mode):
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
            'GI: ' + reason + ' — output grid set from the first frames; '
            're-integrate with explicit 1D/2D ranges if you need the full '
            'incidence extent.')
        logger.info(msg)
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

    def _frame_source_for(self, scan):
        r"""Build a core :class:`FrameSource` over this scan's per-file image
        series for the headless GI freeze prepass (ADR-0006 STEP 2).

        Returns:
          - :class:`ImageFileSource` for a single detector file (``single_img``);
          - :class:`TiffSeriesSource` over the STRICT ``^{scan}_\d+\.{ext}$`` file
            list from :meth:`_enumerate_scan_files` (NOT ``from_directory`` — a
            directory sweep would scoop up neighbour scans / background frames
            that the anchored regex deliberately excludes);
          - ``None`` for Eiger/HDF5 masters and Image-Directory sources that
            cannot be cheaply per-frame swept (the caller maps ``None`` to the
            old empty-``_enumerate_scan_files`` skip-vs-unverifiable split).

        The manifest metadata format is ``self.meta_ext`` so the headless sweep
        reads the EXACT sidecars the GUI does; a falsy meta_ext ⇒ no manifest ⇒
        the source's ``has_scan_manifest`` capability is False ⇒
        ``prepare_gi_freeze`` reports 'unverifiable' (warn-and-proceed)."""
        img_file = getattr(self, "img_file", None)
        if not img_file or _is_eiger_master(img_file):
            return None
        img_ext = getattr(self, "img_ext", "") or ""
        if img_ext.lower() in ('h5', 'hdf5', 'nxs'):
            return None
        if getattr(self, "inp_type", None) == 'Image Directory':
            return None
        meta_fmt = getattr(self, "meta_ext", None) or None
        meta_dir = getattr(self, "meta_dir", None)
        if getattr(self, "single_img", False):
            return ImageFileSource(
                img_file, metadata_format=meta_fmt, meta_dir=meta_dir)
        files = self._enumerate_scan_files()
        if not files:
            return None
        return TiffSeriesSource(
            [fname for fname, _num in files],
            metadata_format=meta_fmt, meta_dir=meta_dir)

    def _gi_whole_scan_scout_entries(self, scan):
        """Decide how to freeze the whole-scan GI grid and, when needed, gather
        the scout images.  Returns ``(status, entries)``:

        - ``("skip", [])`` — fixed/manual angle, Eiger/master, a single-frame
          scan, or a swept scan with one incidence: the existing chunk-local /
          session freeze is correct (the grid was never clipped).
        - ``("freeze", [lo_entry, hi_entry])`` — a varying-incidence per-file
          scan whose global incidence extremes were discovered by core; the
          ``entries`` are ``(img_file, img_number, img_data, img_meta, bg_raw,
          0.0)`` for the lowest+highest incidence frames, images loaded.
        - ``("unverifiable", [])`` — a source that can't be cheaply swept up
          front (Image-Directory, or a possibly-multi-master Eiger): the caller
          warns and proceeds on the first-chunk freeze (T0-4 policy).
        - ``("abort", [])`` — a multi-file scan whose incidence we genuinely
          cannot establish (fewer than two readable incidences across the whole
          series).  The caller also warns and proceeds (T0-4; name predates it).

        ADR-0006 STEP 2: the whole-scan incidence DISCOVERY now lives in
        ``xrd_tools.reduction.prepare_gi_freeze`` (image-free, never raises) over
        a core :class:`FrameSource`; xdart keeps only the scout LOAD + the freeze
        invocation (which can't move to core: chunk 1 can't see the last frame).
        """
        motor = self.incidence_motor
        try:
            float(motor)        # fixed/manual: one angle for the whole scan
            return "skip", []
        except (TypeError, ValueError):
            pass
        source = self._frame_source_for(scan)
        if source is None:
            # _frame_source_for() returns None for sources that can't be cheaply
            # swept per-frame.  Preserve the old skip-vs-unverifiable split:
            #   RISKY ("unverifiable", warn-and-proceed): Image-Directory per-file
            #     GI sweep, or a (possibly multi-master) Eiger master with a
            #     non-fixed motor — the session WOULD clip later frames.
            #   SAFE ("skip"): missing source attrs / no img_file — not an angle
            #     dependence sweep at all (so the grid was never chunk-clipped).
            img_file = getattr(self, "img_file", None) or ""
            if getattr(self, "inp_type", None) == "Image Directory":
                return "unverifiable", []
            if img_file and _is_eiger_master(img_file):
                return "unverifiable", []
            return "skip", []
        # Headless whole-scan incidence DISCOVERY (ADR-0006): core sweeps the
        # source's metadata-only manifest and pins the global incidence extremes
        # into plan.extra["gi_freeze_scout_indices"].  The minimal GI plan is a
        # vehicle: a non-None gi with an unpinned 1D output range so discovery
        # isn't short-circuited (whether THIS run needs a freeze at all is the
        # caller's _gi_ranges_fully_pinned decision, not this method's).  The
        # real incidence motor is passed explicitly.
        plan = ReductionPlan(
            integration_1d=Integration1DPlan(),  # radial_range None => freeze
            gi=GIMode(incidence_motor=str(motor)),
        )
        plan2, diag = prepare_gi_freeze(source, plan, incidence_motor=motor)
        if diag.status == "skip":
            # fixed/single incidence, <2 frames, or one distinct incidence.
            return "skip", []
        if diag.status == "unverifiable":
            # >=2 frames but <2 readable incidences (or no manifest): we cannot
            # establish the global range -> warn-and-proceed (old "abort").
            return "abort", []
        # diag.status == "frozen": load the extreme scouts BY INDEX against the
        # SAME source prepare_gi_freeze swept.  A TiffSeriesSource labels frames
        # by POSITION (1..N) in the strict file list, so the scout indices are
        # positional -- mapping each back through the source recovers the real
        # on-disk file + its scan img_number (non-contiguous / non-1-based
        # filenames stay correct; the contiguous Combi4 fixture would not catch
        # a positional/number confusion on its own).
        indices = plan2.extra.get("gi_freeze_scout_indices") or []
        entries = []
        for idx in indices:
            frame = source.frame_for(int(idx))
            fname = str(frame.source_path) if frame.source_path else ""
            meta = dict(frame.metadata)
            data = np.asarray(source.load_frame(int(idx)), dtype=float)
            _sname, img_number = _get_scan_info(fname)
            # bg is irrelevant to the frozen AXIS extent (geometry+incidence
            # driven), so a failing/missing background must NOT abort an
            # otherwise-valid GI run -- degrade to 0 for the axis-only scout.
            try:
                bg = self.get_background(fname, img_number, meta)
            except Exception:
                logger.debug("GI scout background failed for %s; using 0 for the "
                             "axis-only scout", fname, exc_info=True)
                bg = 0.0
            entries.append((fname, img_number, data, meta, bg, 0.0))
        return "freeze", entries

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
            rf'^{re.escape(scan_name)}[_-]\d+\.{re.escape(img_ext)}$',
            re.IGNORECASE,
        )
        paths = [str(p) for p in _paths_with_suffix(Path(img_dir), f'.{img_ext}')
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
        # Resume (or stop): clear the session's paused flag (4a) so the next
        # adapter.submit() isn't rejected.  No-op without an adapter / when
        # not paused / finished; harmless on the stop path.
        adapter = getattr(self, '_scan_session_adapter', None)
        if adapter is not None:
            adapter.resume()

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
            adapter = getattr(self, '_scan_session_adapter', None)
            session = getattr(self, '_streaming_session', None)
            sink = getattr(self, '_streaming_sink', None)
            scan = self._active_scan
            timeout = getattr(self, 'PAUSE_DRAIN_TIMEOUT', 30.0)
            # Quiesce any open streaming session first (a no-op when nothing
            # is in-flight) so the writer is provably idle before we touch the
            # file from THIS thread.  4c-1: route through the adapter, which
            # delegates to ReductionSession.pause (sets is_paused + drains);
            # _wait_if_paused calls adapter.resume() when the pause ends.
            drained = True
            if adapter is not None:
                drained = adapter.quiesce(timeout=timeout)
            elif session is not None:
                # Defensive no-adapter fallback (unreachable on the streaming
                # path — _get_streaming_session always builds the adapter).
                # Prefer the ReductionSession drain(); a bare public ScanSession
                # only exposes pause() (drain + flag).
                quiesce = (getattr(session, 'drain', None)
                           or getattr(session, 'pause', None))
                drained = quiesce(timeout=timeout) if quiesce is not None else True
            if not drained:
                # RS-1: the writer is provably NOT idle (a stalled worker) — a
                # save/flush from this thread would race the writer's own
                # write()→flush() (single-writer invariant), and resetting the
                # save counter without saving would break persist-before-evict.
                # Pause still proceeds (submits have stopped); the tail flushes
                # on resume/finish.
                logger.warning("pause drain timed out (worker not idle); "
                               "pausing without a flush")
            elif scan is not None and self._frames_since_save > 0:
                # Serial path is the active writer (watch loop / serial
                # dispatch): flush the serial tail.
                if self.xye_only:
                    self._flush_xye_buffer(scan)      # no .nxs save in xye-only
                    self._frames_since_save = 0
                else:
                    self.flush_serial_tail(scan, force=True)
            elif adapter is not None:
                # Streaming path (batch / reprocess / live Phase 2): in-flight
                # window drained (non-terminal — session stays open) + flush
                # via the adapter (h5pool bracket stays in QtNexusSink.flush).
                adapter.flush()
            elif session is not None and sink is not None:
                sink.flush(force=True)         # defensive (no adapter built)
        except Exception:
            # A drain/flush failure must not strand the run; log loudly and
            # still signal the pause (we've stopped submitting, so the writer is
            # idle and disk reads are safe even if the final flush lagged).
            logger.error("error draining/flushing on pause", exc_info=True)
        finally:
            self.sigPaused.emit()

    @contextmanager
    def _h5pool_bracket(self, scan):
        """Pause the shared h5 pool around a serial write to ``scan.data_file``,
        resuming even if the wrapped body raises (the symmetric bracket the .nxs
        single-writer path needs).  Shared by :meth:`flush_serial_tail` and
        ``QtNexusSink.flush`` (the streaming write reuses ONLY this bracket,
        keeping its own ``mode=`` + bookkeeping).

        Callers enter this while holding ``file_lock``.  Readers also acquire
        ``file_lock`` before borrowing from the pool, so pausing under that lock
        closes only idle cached handles, never a handle in active use.
        """
        _get_h5pool().pause(scan.data_file)
        try:
            yield
        finally:
            _get_h5pool().resume(scan.data_file)

    def flush_serial_tail(self, scan, *, force=False) -> bool:
        """The serial save tail (the DRYed copy-paste idiom).

        When a save is due (cadence / cap-pressure / ``force``), h5pool-bracket a
        file-locked ``scan._save_to_nexus()``, drain the XYE buffer, and reset
        the per-save counter.  Returns ``True`` iff it saved.

        persist-before-evict: ``_save_to_nexus`` marks the written frames
        persisted BEFORE ``_frames_since_save`` is reset (the counter that gates
        the next save cycle), so an unsaved frame is never evicted.  ``force=True``
        respects the per-site ``_frames_since_save > 0`` precondition because
        ``_save_due(force=True)`` is False on an empty tail.  No-op in xye_only
        mode (no .nxs target — ``_save_due`` returns False) and for ``scan is
        None``.
        """
        if scan is None or not self._save_due(scan, force=force):
            return False
        with self.file_lock:
            with self._h5pool_bracket(scan):
                scan._save_to_nexus()
        self._flush_xye_buffer(scan)
        self._frames_since_save = 0
        return True

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
        if self.xye_only:
            return False
        # Phase 4b-3: the same headless FlushPolicy the streaming sink uses
        # (kills the predicate divergence).  Serial owns the LIVE unsaved
        # count, so it passes it through; the cap−margin pressure bound (the
        # data-loss fix) and the LIVE_SAVE_INTERVAL upper bound are both the
        # policy's now.  margin defaults to 8 (the canonical value, == the
        # sink's _SAVE_BEFORE_EVICT_MARGIN).
        cap = getattr(scan.frames, "_in_memory_cap", 64)
        counter = getattr(scan.frames, "unsaved_in_memory_count", None)
        unsaved = counter() if callable(counter) else None
        policy = FlushPolicy(interval=self.LIVE_SAVE_INTERVAL, cap=cap)
        return policy.should_flush(frames_since_flush=self._frames_since_save,
                                   unsaved_in_memory=unsaved, force=force)

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

        _n_save = self._frames_since_save        # reset by the tail
        _t_save0 = time.time()
        if self.flush_serial_tail(scan, force=force_save):
            logger.info(
                '[SAVE] %d frames since last save  flush=%.3fs',
                _n_save, time.time() - _t_save0,
            )
        elif self.xye_only:
            # Int 1D (XYE) on the serial fallback: there is no .nxs save to
            # ride on (_save_due is always False in this mode), so drain the
            # XYE buffer per dispatch -- without this the serial path wrote
            # ZERO output, silently.  (The default streaming path flushes
            # via QtNexusSink.flush.)
            self._flush_xye_buffer(scan)
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
        pending = imageThread._series_average_pending(self, pending)
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
        pending = imageThread._series_average_pending(self, pending)
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
        adapter = self._scan_session_adapter
        count = 0
        for live in frames:
            self._wait_if_paused()        # pause between frames (streaming path)
            if self.command == 'stop':
                break
            # Per-frame status is emitted at COMPLETION by the sink
            # (QtNexusSink._emit_frame_status) so the label tracks what the
            # plots show -- emitting here at submit time raced frames ahead
            # of the display in the parallel pipeline.
            #
            # 4c-1: register+submit + the stop-on-write-failure translation
            # (which must NOT raise out of run(), or it tears down the
            # QThread) live in the adapter; it returns False on failure
            # after setting command='stop'.
            if not adapter.submit(live):
                break
            count += 1
        return count

    def _on_qt_gui_thread(self) -> bool:
        try:
            app = Qt.QtWidgets.QApplication.instance()
            return app is not None and Qt.QtCore.QThread.currentThread() is app.thread()
        except Exception:
            return False

    def _record_store_hydrator(self, scan, record_store):
        """Build the disk-backed ``FrameRecordStore`` hydrator.

        This hydrator is registered on the session store but is intentionally
        invoked by ``FrameHydrationWorker``.  Refusing the Qt GUI thread here
        protects the live UI if a caller accidentally asks the store to hydrate
        synchronously from a render path.
        """
        initial_path = getattr(scan, "data_file", None)

        def hydrate(label):
            thread_check = getattr(self, "_on_qt_gui_thread", None)
            on_gui_thread = (
                thread_check()
                if callable(thread_check)
                else imageThread._on_qt_gui_thread(self)
            )
            if on_gui_thread:
                logger.error(
                    "FrameRecordStore hydrator refused GUI-thread disk read "
                    "for frame %s", label)
                return None
            scan_file = getattr(scan, "data_file", None) or initial_path
            if not scan_file:
                return None
            try:
                frame_label = int(label)
            except (TypeError, ValueError):
                return None

            existing = record_store.get(frame_label)
            mode_1d = (
                existing.active_mode_1d
                if existing is not None and existing.results_1d
                else DEFAULT_MODE_KEY
            )
            mode_2d = (
                existing.active_mode_2d
                if existing is not None and existing.results_2d
                else DEFAULT_MODE_KEY
            )
            try:
                from xrd_tools.io import read_frame_view
                file_lock = (
                    getattr(self, "file_lock", None)
                    or getattr(scan, "file_lock", None)
                )
                if file_lock is None:
                    view = read_frame_view(
                        scan_file, frame_label,
                        mode_1d=mode_1d, mode_2d=mode_2d)
                else:
                    # Same lock the serial and streaming writers hold around
                    # _save_to_nexus, so background hydration can proceed during a
                    # pause but cannot overlap the writer's HDF5 r+/w open.
                    with file_lock:
                        view = read_frame_view(
                            scan_file, frame_label,
                            mode_1d=mode_1d, mode_2d=mode_2d)
            except Exception:
                logger.debug("record-store hydrate failed for %s", label,
                             exc_info=True)
                return None
            return FrameRecord.from_view(
                view, mode_1d=mode_1d, mode_2d=mode_2d)

        return hydrate

    def _heavy_staging_window(self, scan):
        """MEM-2: the RAM-aware size for the live heavy caps.

        One number, consumed by all three heavy caps (LiveFrameSeries staging,
        FrameRecordStore + PublicationStore ``max_heavy_items``).  ~25% of TOTAL
        physical RAM / the as-stored per-frame heavy cost (raw is float64-upcast
        today → 8 B/px), clamped [16, 64]; env ``XDART_HEAVY_WINDOW`` pins it.
        Computed at run start (detector shape known); cached + logged once.
        """
        cached = getattr(self, "_heavy_window", None)
        if cached is not None:
            return cached
        from xrd_tools.core import heavy_window, heavy_window_log_line
        shape = getattr(self, "detector_shape", None)
        frame_bytes = None
        if shape and len(shape) >= 2:
            frame_bytes = int(shape[0]) * int(shape[1]) * 8   # float64 as-stored
        window = heavy_window(frame_bytes)
        self._heavy_window = window
        logger.info(heavy_window_log_line(
            window, frame_bytes,
            overridden=bool(os.environ.get("XDART_HEAVY_WINDOW"))))
        return window

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
        # MEM-3: map Cores honestly to the reduction pool, capped at the
        # throughput knee (memory-aware) and NEVER to None — the old
        # ``n_workers==1 -> None`` silently built a ~20-worker default pool
        # (20 integrator deepcopies), the opposite of the requested serial run.
        from xrd_tools.core import (
            reduction_worker_cap, reduction_worker_cap_log_line)
        executor = reduction_worker_cap(self.max_cores)
        logger.info(reduction_worker_cap_log_line(
            executor, requested=self.max_cores,
            overridden=bool(os.environ.get("XDART_REDUCTION_WORKERS"))))
        cancel_token = self._cancel_token() if hasattr(self, "_cancel_token") else None
        # Batch brackets all frames (scout_union over first+last of the chunk);
        # live has no last frame at session open, so it freezes from the first
        # frame only (matches the legacy _process_one live path + the #75 advisory).
        _default_freeze = ("scout_union" if self.batch_mode else "first_frame") \
            if self.gi else None
        gi_freeze_mode = getattr(self, "gi_freeze_mode", _default_freeze)
        # MEM-2: RAM-aware heavy window feeds all three heavy caps.  Set the
        # LiveFrameSeries staging cap here so the record store (which mirrors it)
        # and the GUI's PublicationStore (which reads self._heavy_window) all
        # take the same value.  Defensive getattr: reduction-session tests drive
        # this through a minimal host double without the wrangler's method — fall
        # back to the pure RAM-aware default (MEM-3 fix for a MEM-2 test gap).
        _hsw = getattr(self, "_heavy_staging_window", None)
        if callable(_hsw):
            window = _hsw(scan)
        else:
            from xrd_tools.core import heavy_window
            window = heavy_window()
        frames_obj = getattr(scan, "frames", None)
        if frames_obj is not None:
            frames_obj._in_memory_cap = window
        record_store = FrameRecordStore(
            max_items=_LIVE_RECORD_STORE_MAX_ITEMS,
            max_heavy_items=window,
        )
        self._streaming_record_store = record_store
        sink = QtNexusSink(
            self, scan, standard_plan, mask=self.mask, record_store=record_store
        )
        try:
            # 4f-bridge: drive the streaming write path through the PUBLIC
            # xrd_tools.session.ScanSession (commands in / events out), not a raw
            # ReductionSession — xdart is now thin over the headless session.
            # It builds + arms its own streaming ReductionSession internally;
            # clear_frame_images=True preserves the PERF-3 raw-nulling, and the
            # QtNexusSink still drives the per-frame display publish (the session
            # forwards every sink hook via its internal _EventSink).
            session = open_live_scan_session(
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
                record_store=record_store,
                record_store_persisted_on_write=False,
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
            self._streaming_record_store = None
            self._scan_session_adapter = None
            return None, None
        self._streaming_session = session
        self._streaming_sink = sink
        self._streaming_scan_id = id(scan)
        # 4c-1: the per-frame submit + pause quiesce/flush route through one
        # adapter (it owns the stop-on-write-failure translation + delegates
        # quiesce to ReductionSession.pause).
        from .scan_session import ScanSessionAdapter
        self._scan_session_adapter = ScanSessionAdapter(self, scan, session, sink)
        self._scan_session_adapter.set_hydrator(
            imageThread._record_store_hydrator(self, scan, record_store))
        return session, sink

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
                f'{self._middle_truncate(fname)} [frame {img_number}]'
            )
        else:
            self.showLabel.emit(self._middle_truncate(fname))

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
        # Wave 5: live scan display is published through _published_frames and
        # consumed by static_scan_widget.update_data into PublicationStore.
        _t_h5_total = _t_h5_wait = _t_h5_write = 0.0

        # ── In-memory accumulation (no disk I/O — batch flush handles that) ──
        if not self.xye_only:
            # Set source file as relative path from HDF5 dir for NeXus provenance
            if img_file:
                frame.source_file = os.path.abspath(str(img_file))
            else:
                frame.source_file = ""
            # source_frame_idx: per-source-file 0-based frame offset.
            # See the matching block in ``_build_batch_frames`` (the streaming
            # dispatcher's frame-shell builder) for the full rationale.  Eiger /
            # HDF5 masters get ``img_number - 1``; everything else stays at 0.
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

            # NOTE: no PERF-3 raw-free here.  This live/non-batch path hands the
            # frame to the GUI publication store; freeing map_raw before that
            # hand-off would force thumbnail/raw rehydration on the hot path.
            # Batch still frees raw in QtNexusSink.worker_process after its
            # publication payload has been prepared.

        # ── XYE buffer (flushed at end of batch by the dispatcher) ──────
        _t5 = time.time()
        with self._xye_lock:
            self._xye_buffer.append((img_number, frame))
        _t_csv = time.time() - _t5

        _t_total = t_read + _t_frame + _t_1d + _t_2d + _t_h5_total + _t_csv
        _perf = getattr(self, '_perf', None)
        if _perf is not None:
            _perf['frame'] += _t_frame
            _perf['1d'] += _t_1d
            _perf['2d'] += _t_2d
            _perf['h5'] += _t_h5_total
            _perf['csv'] += _t_csv
            _perf['n'] += 1
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

    def _read_eiger_metadata(self, master_path):
        """Read per-master metadata once, returning a per-frame copy."""
        if not self.meta_ext:
            return {}
        key = (
            os.path.abspath(str(master_path)),
            str(self.meta_ext),
            os.path.abspath(str(self.meta_dir)) if self.meta_dir else '',
        )
        if key not in self._eiger_metadata_cache:
            self._eiger_metadata_cache[key] = read_image_metadata(
                master_path, meta_format=self.meta_ext, meta_dir=self.meta_dir,
            )
        return dict(self._eiger_metadata_cache[key])

    @staticmethod
    def _eiger_scan_name(master_path):
        master_stem = Path(master_path).stem
        return master_stem[:-7] if master_stem.lower().endswith('_master') else master_stem

    def _eiger_refill_master_queue(self):
        """Queue matching HDF5 master / NeXus files not yet processed."""
        match = _name_filter(self.file_filter)
        img_ext = (getattr(self, 'img_ext', '') or '').lower().lstrip('.')
        if not img_ext:
            return
        suffix = f'_master.{img_ext}' if img_ext in ('h5', 'hdf5') else f'.{img_ext}'
        candidates = _paths_with_suffix(
            Path(self.img_dir), suffix, recursive=self.include_subdir)
        master_files = sorted(
            p for p in candidates if match(p.name[:-len(suffix)])
        )
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
                    scan_name = self._eiger_scan_name(self._eiger_master_path)

                    # Advance the shared frame cursor *before* dispatching so
                    # that any concurrent sync read (e.g. in fallback) does
                    # not re-serve these frames.
                    self._eiger_frame_idx = end
                    to_read = []
                    for frame_idx in range(start, end):
                        self._record_discovered_frame()
                        if self._should_skip_before_read(scan_name, frame_idx + 1):
                            continue
                        to_read.append(frame_idx)

                    if not to_read:
                        continue

                    meta = self._read_eiger_metadata(self._eiger_master_path)
                    groups = []
                    for frame_idx in to_read:
                        if not groups or frame_idx != groups[-1][1]:
                            groups.append([frame_idx, frame_idx + 1])
                        else:
                            groups[-1][1] = frame_idx + 1

                    bulk_failed = False
                    for group_start, group_end in groups:
                        _t_blk = time.time()
                        try:
                            block = np.asarray(
                                self._eiger_h5_dataset[group_start:group_end])
                        except Exception as e:
                            logger.warning(
                                'Bulk read failed (start=%d end=%d): %s; '
                                'falling back to per-frame read',
                                group_start, group_end, e,
                            )
                            self._eiger_frame_idx = group_start
                            bulk_failed = True
                            break  # outer worker loop resumes with sync reader
                        _t_blk = time.time() - _t_blk
                        _perf = getattr(self, '_perf', None)
                        if _perf is not None:
                            _perf['prefetch_io'] += _t_blk
                            _perf['prefetch_frames'] += (group_end - group_start)
                        # If a bulk read takes >50ms it can fight with the
                        # consumer for memory bandwidth — log so we can
                        # correlate against [TIMING] spikes on the consumer.
                        if _t_blk > 0.05:
                            logger.info(
                                '[PREFETCH] block frames %d-%d read in %.3fs',
                                group_start, group_end - 1, _t_blk,
                            )

                        for offset, frame_idx in enumerate(
                                range(group_start, group_end)):
                            if (self._prefetch_stop_evt.is_set()
                                    or self.command == 'stop'):
                                return
                            item = (
                                self._eiger_master_path,
                                scan_name,
                                frame_idx + 1,   # 1-based img_number
                                block[offset],
                                meta,
                            )
                            if not self._push_frame_to_queue(item):
                                return
                    if bulk_failed:
                        break
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
        while True:
            # ── Initialise on the very first call ────────────────────────────
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

            # ── Current master exhausted?  Try to advance ────────────────────
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

            # ── Read one frame ────────────────────────────────────────────────
            frame_idx = self._eiger_frame_idx
            self._eiger_frame_idx += 1
            scan_name = self._eiger_scan_name(self._eiger_master_path)
            img_number = frame_idx + 1  # 1-based
            self._record_discovered_frame()
            if self._should_skip_before_read(scan_name, img_number):
                continue

            try:
                if self._eiger_fabio_handle is not None:
                    # Primary: persistent fabio handle
                    _raw = (self._eiger_fabio_handle.data if frame_idx == 0
                            else self._eiger_fabio_handle.get_frame(frame_idx).data)
                    img_data = np.asarray(_raw)
                elif self._eiger_h5_dataset is not None:
                    # Fallback: h5py dataset
                    img_data = np.asarray(self._eiger_h5_dataset[frame_idx])
                else:
                    # Should not happen, but safety net
                    with fabio.open(self._eiger_master_path) as _img:
                        _raw = _img.data if frame_idx == 0 else _img.get_frame(frame_idx).data
                    img_data = np.asarray(_raw)
            except Exception as e:
                logger.error('Error reading frame %d from %s: %s', frame_idx, self._eiger_master_path, e)
                self._record_skip_reason("unreadable or empty image data")
                img_data = None

            meta = self._read_eiger_metadata(self._eiger_master_path)

            return self._eiger_master_path, scan_name, img_number, img_data, meta

    # ── Image iteration ──────────────────────────────────────────────────

    def get_next_image(self):
        """Gets next image in image series or in directory to process."""
        is_master = _is_eiger_master(self.img_file) if self.img_file else False

        if self.single_img and not is_master:
            scan_name, img_number = _get_scan_info(self.img_file)
            self._record_discovered_frame()
            if self._should_skip_before_read(scan_name, img_number):
                self.sigUpdate.emit(self._append_output_number(img_number))
                return None, scan_name, img_number, None, {}
            img_data = np.asarray(read_image(self.img_file), dtype=float)
            meta = read_image_metadata(self.img_file, meta_format=self.meta_ext, meta_dir=self.meta_dir) if self.meta_ext else {}
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
                    rf'^{re.escape(self.scan_name)}[_-]\d+\.{re.escape(self.img_ext)}$',
                    re.IGNORECASE,
                )
                self.img_fnames = [
                    p for p in _paths_with_suffix(Path(self.img_dir), f'.{self.img_ext}')
                    if _series_re.match(p.name)
                ]
            else:
                first_img = ''
                match = _name_filter(self.file_filter)
                suffix = f'.{self.img_ext}'
                candidates = _paths_with_suffix(
                    Path(self.img_dir), suffix, recursive=self.include_subdir)
                self.img_fnames = (p for p in candidates
                                   if match(p.name[:-len(suffix)]))

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
            self._record_discovered_frame()

            if self._should_skip_before_read(sname, snumber):
                continue

            data = np.asarray(read_image(fname), dtype=float)
            if data is None or not np.isfinite(data).any():
                self._record_skip_reason("unreadable or empty image data")
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
        if not self.meta_ext:
            # GUI "none"/blank is an off switch.  The reusable headless reader
            # maps None to auto, so the GUI worker must guard before calling it.
            return {}
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
                          detector_shape=self.detector_shape,
                          # J2: share lock with wrangler save path
                          file_lock=self.file_lock,
                          **self.scan_args)
        scan.skip_2d = self.scan.skip_2d
        # N1: the project root -> entry/@source_base + relative raw source paths
        # in the writer (portable .nxs).  None -> absolute paths (back-compat).
        scan.source_base = getattr(self, "source_base", None)
        # v2 NeXus writer needs a Diffractometer to derive per-frame
        # rot1/rot2/rot3 + incidence-angle arrays from scan_data.  default_geometry()
        # picks the preset from what scan_data recorded: psic when nu/del are
        # present (RSM/6-circle), else the two-circle convention (rot1←tth,
        # incidence ← the resolved sample-tilt motor).  Override later from the
        # geometry UI panel when the user picks a different convention.
        scan.default_geometry()

        write_mode = self.write_mode
        if not os.path.exists(fname):
            write_mode = 'Overwrite'

        # Int 1D (XYE) writes ONLY .xye files (via the XYE flush); it must not
        # create or write the .nxs stack at all.  Skip all NeXus disk I/O here —
        # the per-batch ``scan._save_to_nexus`` is already gated off for
        # xye_only, so the scan object is used purely in-memory for integration.
        if not self.xye_only:
            with self.file_lock:
                with self._h5pool_bracket(scan):
                    if write_mode == 'Append':
                        # v2 NeXus loader (the only one we support now).
                        scan.load_from_h5(replace=False, mode='r')
                        scan.skip_2d = self.scan.skip_2d
                        for (k, v) in self.scan_args.items():
                            setattr(scan, k, v)
                        self._remember_append_skip_snapshot(scan.name, scan=scan)
                        existing_frames = scan.frames.index
                        if len(existing_frames) == 0:
                            scan.save_to_nexus(replace=True)
                    else:
                        scan.save_to_nexus(replace=True)

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
                bg_meta = self.get_meta_data(bg_file)
        elif self.bg_type == 'Series Average':
            if self.bg_file:
                sname, fnames, bg, bg_meta = get_series_avg(self.bg_file, self.detector, self.meta_ext)
                if sname is None:
                    return 0
        else:
            if self.bg_dir and (self.bg_match_fname or self.bg_matching_par):
                bg_file_filter = 'bg' if not self.bg_file_filter else self.bg_file_filter
                match = _name_filter(bg_file_filter)
                if self.bg_match_fname:
                    # scan_name is DATA, not filter grammar: a name starting
                    # with '-' or containing '|'/'OR' must stay a literal
                    # substring requirement, so conjoin it outside the
                    # compiled expression.
                    scan_term = str(self.scan_name).lower()
                    match = (lambda name, _m=match, _t=scan_term:
                             _t in str(name).lower() and _m(name))
                if not self.meta_ext:
                    meta_files = []
                    suffix = ''
                else:
                    suffix = f'.{self.meta_ext}'
                    meta_files = sorted(
                        str(f) for f in _paths_with_suffix(self.img_dir, suffix)
                        if match(f.name[:-len(suffix)])
                    )

                for meta_file in meta_files:
                    bg_file = f'{os.path.splitext(meta_file)[0]}.{self.img_ext}'
                    if bg_file == img_file:
                        bg_file = None
                        continue

                    bg_meta = self.get_meta_data(bg_file)
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
