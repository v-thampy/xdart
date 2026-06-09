# -*- coding: utf-8 -*-
"""
@author: walroth
"""

# Standard library imports
import logging
import os
from queue import Queue
import threading
import traceback

# Other imports
import numpy as np
from pathlib import Path

# Qt imports
from pyqtgraph import Qt
from pyqtgraph.parametertree import Parameter

# This module imports
from xdart.utils.h5pool import get_pool as _get_h5pool
from ssrl_xrd_tools.io.export import write_xye

logger = logging.getLogger(__name__)


# Sentinel used by ``_apply_threshold_inline`` to mark out-of-band
# pixels.  pyFAI's CSR integrator auto-skips NaN at integrate time
# without invalidating the mask CRC, so the per-frame threshold
# filter survives without forcing a per-frame LUT rebuild.  See the
# session_may2026_nexusformat_writer.md lesson notes for the full
# story (lessons 2, 3 in particular).
_THRESHOLD_NAN = np.float32(np.nan)

# Default cadence: flush scan state to disk every N frames in batch
# / live modes.  Subclasses can override per-instance via the
# ``LIVE_SAVE_INTERVAL`` attribute if they want a different rhythm.
_LIVE_SAVE_INTERVAL = 8
# 1D-only .nxs writes are small and cheap, so flush far less often: it is the
# fixed per-save overhead (not per-frame compute) that made long Int-1D scans
# crawl as the frame count grew.  2D keeps the tight default so peak RAM stays
# bounded.  (PERF-2)
# This is now an UPPER bound on save spacing, not the effective cadence: the
# persist-before-evict fix (LiveFrameSeries._persisted + mark_persisted, and the
# _save_due cap bound in imageWranglerThread) guarantees a save fires before the
# unsaved in-memory set reaches _in_memory_cap, so no frame's int_1d is ever
# evicted before it's written — the high interval is safe on scans longer than
# the cap.  Effective cadence is therefore min(this, cap-margin).  See
# review/CC_data_loss_save_vs_evict_jun2026.md.
_LIVE_SAVE_INTERVAL_1D = 1000


class _CommandCancelToken:
    """Duck-typed ssrl CancelToken bound to a wrangler thread command."""

    def __init__(self, owner):
        self._owner = owner

    @property
    def cancelled(self):
        return getattr(self._owner, 'command', None) == 'stop'


class wranglerWidget(Qt.QtWidgets.QWidget):
    """Base class for wranglers. Extending this ensures all methods,
    signals, and attributes expected by ttheta_widget are present.
    Threads should be started by use of sigStart.emit, which ensures
    tthetaWidget handles initiation.
    
    attributes:
        command_queue: Queue, used to send commands to thread
        file_lock, mp.Condition, process safe lock for file access
        fname: str, path to data file
        parameters: pyqtgraph Parameter, stores parameters from user
        scan_name: str, current scan name, used to handle syncing data
        scan_args: dict, used as **kwargs in scan initialization.
            see LiveScan.
        thread: wranglerThread or subclass, QThread for controlling
            processes
    
    methods:
        enabled: Enables or disables interactivity
        set_fname: Method to safely change file name
        setup: Syncs thread parameters prior to starting
    
    signals:
        finished: Should be connected to thread.finished signal
        sigStart: Tells tthetaWidget to start the thread and prepare
            for new data.
        sigUpdateData: int, signals a new frame has been added.
        sigUpdateFile: (str, str, bool, str, bool, bool), sends new scan_name, file name
            GI flag (grazing incidence), theta motor for GI, single_image and
            series_average flag to static_scan_Widget.
    """
    sigStart = Qt.QtCore.Signal()
    sigUpdateData = Qt.QtCore.Signal(int)
    # sigUpdateFrame = Qt.QtCore.Signal(dict)
    sigUpdateFile = Qt.QtCore.Signal(str, str, bool, str, bool, bool)
    sigUpdateGI = Qt.QtCore.Signal(bool)
    finished = Qt.QtCore.Signal()
    started = Qt.QtCore.Signal()

    def __init__(self, fname, file_lock, parent=None):
        """fname: str, file path
        file_lock: mp.Condition, process safe lock
        """
        super().__init__(parent)
        self.file_lock = file_lock
        self.fname = fname
        self.scan_name = 'null_thread'
        self.parameters = Parameter.create(
            name='wrangler_widget', type='int', value=0
        )
        self.scan_args = {}

        self.command_queue = Queue()
        self.thread = wranglerThread(self.command_queue, self.scan_args, self.fname, self.file_lock, self)
        self.thread.finished.connect(self.finished.emit)
        self.thread.started.connect(self.started.emit)
        self.thread.sigUpdate.connect(self.sigUpdateData.emit)
        # self.thread.sigUpdateFrame.connect(self.sigUpdateFrame.emit)
        self.thread.sigUpdateGI.connect(self.sigUpdateGI.emit)

    def enabled(self, enable):
        """Use this function to control what is enabled and disabled
        during integration.
        """
        pass

    def setup(self):
        """Sets the thread child object. Called by tthetaWidget prior
        to starting thread.
        """
        # Disconnect old thread signals to avoid duplicate emissions
        try:
            self.thread.finished.disconnect(self.finished.emit)
            self.thread.started.disconnect(self.started.emit)
            self.thread.sigUpdate.disconnect(self.sigUpdateData.emit)
            self.thread.sigUpdateGI.disconnect(self.sigUpdateGI.emit)
        except (TypeError, RuntimeError):
            pass  # Signals were never connected or already disconnected
        self.thread = wranglerThread(self.command_queue, self.scan_args, self.fname, self.file_lock, self)
        self.thread.finished.connect(self.finished.emit)
        self.thread.started.connect(self.started.emit)
        self.thread.sigUpdate.connect(self.sigUpdateData.emit)
        self.thread.sigUpdateGI.connect(self.sigUpdateGI.emit)

    def set_fname(self, fname):
        """Changes fname attribute of self and thread.
        args:
            fname: str, path for new file.
        """
        with self.file_lock:
            if not self.thread.isRunning():
                self.fname = fname
                self.thread.fname = fname


class wranglerThread(Qt.QtCore.QThread):
    """Base class for wranglerThreads. Used to manage processes
    including data and command queues. Subclasses should override the
    run method.
    
    attributes:
        command_q: mp.Queue, queue to send commands to process
        file_lock: mp.Condition, process safe lock for file access
        fname: str, path to data file.
        input_q: mp.Queue, queue for commands sent from parent
        signal_q: mp.Queue, queue for commands sent from process
        scan_args: dict, used as **kwargs in scan initialization.
            see LiveScan.
    
    methods:
        run: Called by start, main thread task.
    
    signals:
        sigUpdate: int, signals a new frame has been added.
        sigUpdateFile: (str, str, bool, str, bool, bool), sends new scan_name, file name
            GI flag (grazing incidence), theta motor for GI, single_image and
            series_average flag to static_scan_Widget.
        sigUpdateGI: bool, signals the grazing incidence condition has changed.
    """
    sigUpdate = Qt.QtCore.Signal(int)
    # sigUpdateFrame = Qt.QtCore.Signal(dict)
    sigUpdateFile = Qt.QtCore.Signal(str, str, bool, str, bool, bool)
    sigUpdateGI = Qt.QtCore.Signal(bool)

    # Save cadence (frames between disk flushes), mode-aware: a 1D-only run
    # (``scan.skip_2d``) flushes every ``_LIVE_SAVE_INTERVAL_1D`` frames; a 2D
    # run keeps the tight ``_LIVE_SAVE_INTERVAL`` for bounded RAM.  An instance
    # may still pin a value (e.g. a test) by assigning ``LIVE_SAVE_INTERVAL``.
    @property
    def LIVE_SAVE_INTERVAL(self) -> int:
        override = getattr(self, "_live_save_interval_override", None)
        if override is not None:
            return int(override)
        if getattr(getattr(self, "scan", None), "skip_2d", False):
            return _LIVE_SAVE_INTERVAL_1D
        return _LIVE_SAVE_INTERVAL

    @LIVE_SAVE_INTERVAL.setter
    def LIVE_SAVE_INTERVAL(self, value: int) -> None:
        self._live_save_interval_override = int(value)

    def __init__(self, command_queue, scan_args, fname, file_lock,
                 parent=None):
        """command_queue: mp.Queue, queue for commands sent from parent
        scan_args: dict, used as **kwargs in scan initialization.
            see LiveScan.
        fname: str, path to data file.
        file_lock: mp.Condition, process safe lock for file access
        """
        super().__init__(parent)
        self.input_q = command_queue # thread queue
        self.scan_args = scan_args
        self.fname = fname
        self.file_lock = file_lock
        self.signal_q = Queue()
        self.command_q = Queue()

        # ── Shared batch-engine state ────────────────────────────────
        # Subclasses can override any of these before .start() (or
        # via their own __init__) — the defaults are the "no batch
        # features active" zero state.

        # XYE write buffer + lock.  Populated during integration in
        # workers; drained at end of batch by _flush_xye_buffer.
        self._xye_buffer: list = []
        self._xye_lock = threading.Lock()

        # Per-batch save cadence counter.  Wraps to zero each time
        # _save_to_disk fires.
        self._frames_since_save = 0

        # In-memory hand-off of just-integrated frames to the main
        # thread so it doesn't have to round-trip through disk.  The
        # main thread's update_data consumes this dict.
        self._published_frames: dict = {}

        # Threshold filtering — subclass sets these from its UI; the
        # base default is "no threshold" so nexus / other wranglers
        # that don't expose a threshold UI pay nothing.
        self.apply_threshold = False
        self.threshold_min = 0
        self.threshold_max = 0

        # Sub-label appended to log lines (e.g. "[Subtracted bg.tif]"
        # for SPEC bg-subtraction mode).  Empty string = no append.
        self.sub_label = ''

        # Mode flags read by the dispatch loops + the GUI's
        # wrangler_finished handler.
        self.batch_mode = False
        self.xye_only = False
        self.max_cores = 1
        self._reduction_session = None
        self._reduction_session_key = None
        # Streaming (PERF-4b) session + its QtNexusSink, kept on dedicated slots
        # because one persistent session spans the WHOLE scan (the chunked cache
        # keys on per-chunk n_workers, which varies).  Finished at scan end by
        # _close_reduction_session.
        self._streaming_session = None
        self._streaming_sink = None
        self._streaming_scan_id = None
        # Set by _close_reduction_session when a streaming write/sink failure
        # surfaces from finish() — so the run can't report a false "success".
        self._reduction_write_error = None

    def run(self):
        """Main task. Subclasses (e.g. imageThread) override this."""
        pass

    # ── Shared batch helpers ────────────────────────────────────────

    def _cancel_token(self):
        return _CommandCancelToken(self)

    def _get_reduction_session(self, key, factory):
        """Return the persistent headless reduction session for *key*.

        The session owns the executor and per-thread pyFAI integrators for the
        scan/run lifetime.  The caller supplies a key that includes scan identity
        and execution policy; changing either closes the old session and opens a
        fresh one from the provided factory.
        """
        if self._reduction_session is not None and self._reduction_session_key == key:
            return self._reduction_session

        self._close_reduction_session()
        self._reduction_session = factory()
        self._reduction_session_key = key
        return self._reduction_session

    def _reduction_session_key_for(self, scan, plan, n_workers):
        try:
            n_workers = int(n_workers or 1)
        except (TypeError, ValueError):
            n_workers = 1
        return (
            id(scan),
            str(getattr(scan, "name", "scan")),
            str(getattr(scan, "data_file", "")),
            max(1, n_workers),
            bool(getattr(scan, "gi", False)),
            bool(getattr(scan, "skip_2d", False)),
            id(plan),
        )

    def _close_reduction_session(self):
        session = self._reduction_session
        self._reduction_session = None
        self._reduction_session_key = None
        # The streaming session's finish() drains the writer thread + does the
        # final QtNexusSink flush (save + XYE + end-of-run signal), so closing
        # it here is the streaming batch's end-of-scan write.
        streaming = self._streaming_session
        self._streaming_session = None
        self._streaming_sink = None
        self._streaming_scan_id = None
        # BLOCKER 2: finish() is fail-loud — a streaming sink/write failure now
        # RAISES instead of being silently swallowed (the user must not think a
        # failed write succeeded).  Close BOTH sessions even if the first raises
        # (wrap each individually + collect), then surface the failure loudly.
        errors = []
        for sess in (session, streaming):
            if sess is not None:
                try:
                    sess.finish()
                except Exception as exc:
                    errors.append(exc)
                    logger.error("reduction session WRITE FAILED on close: %s",
                                 exc, exc_info=True)
        if errors:
            self._reduction_write_error = errors[0]
            msg = (f"Save FAILED — output .nxs may be incomplete: {errors[0]}")
            show = getattr(self, "showLabel", None)
            if show is not None:
                try:
                    show.emit(msg)
                except Exception:
                    pass
            # A failed write is serious — stop the run rather than process
            # further scans onto a broken output.
            if getattr(self, "command", None) is not None:
                self.command = 'stop'

    def _resolve_frame_mask(self, scan, img_data):
        """Return a stable per-scan "bad pixel" mask cached on the scan.

        Computed once from ``img_data < 0`` of the first frame seen
        by this scan; reused for every subsequent frame.  Keeping
        the mask stable across frames is what lets pyFAI's CSR
        engine cache stay valid — a single pixel changing in the
        mask invalidates the cache and forces a ~250 ms LUT rebuild
        (observed on Eiger scans where saturation flicker shifts the
        mask CRC frame-to-frame; see session_may2026 lesson 1).

        Per-frame threshold filtering is NOT routed through this
        mask — see :meth:`_apply_threshold_inline` for that path
        (NaN-sentinel in the data, mask CRC unchanged).

        F3: callers in the parallel section should pre-warm the
        cache via :meth:`_prewarm_frame_mask` on the main thread
        BEFORE submitting work, so the cache is fully populated
        when N workers read it.  Without the prewarm, the first N
        workers all see ``None`` and race to write the same value —
        currently safe because every worker computes the SAME mask,
        but the invariant isn't enforced by the code and a future
        change (e.g. per-worker thresholding) could break it.
        """
        cached = getattr(scan, '_cached_data_mask', None)
        if cached is None:
            try:
                cached = np.arange(img_data.size)[
                    np.asarray(img_data).flatten() < 0
                ]
            except (AttributeError, TypeError, ValueError) as e:
                logger.debug("frame-mask compute failed: %s", e)
                cached = None
            scan._cached_data_mask = cached
        return cached

    def _prewarm_frame_mask(self, scan, img_data) -> None:
        """Populate ``scan._cached_data_mask`` on the main thread.

        F3 — prevents the racy initialization that happens when N
        parallel workers all simultaneously see a ``None`` cache and
        each compute + write the same mask.  Computing on the main
        thread before submitting any worker means every worker only
        ever does a cache *read* against a stable value.

        Idempotent: a no-op when the cache is already set.  Called
        from each wrangler's run loop with the first frame's
        ``img_data`` before the parallel section.
        """
        if getattr(scan, '_cached_data_mask', None) is not None:
            return
        self._resolve_frame_mask(scan, img_data)

    def _apply_threshold_inline(self, img_data):
        """Pre-clamp pixels outside the threshold band to NaN.

        Returns a fresh float32 array with out-of-band pixels
        replaced by NaN.  pyFAI's CSR integrator skips NaN pixels
        automatically (no ``dummy``/``delta_dummy`` kwargs needed),
        and NaN propagates cleanly through bg subtraction and monitor
        normalization arithmetic inside ``frame.integrate_1d/2d``.

        No-op when ``self.apply_threshold`` is False — subclasses
        that don't expose a threshold UI inherit a free pass-through.
        """
        if not self.apply_threshold:
            return img_data
        img = np.asarray(img_data, dtype=np.float32, copy=True)
        bad = (img < self.threshold_min) | (img > self.threshold_max)
        img[bad] = _THRESHOLD_NAN
        return img

    def _flush_xye_buffer(self, scan, published_idxs=None):
        """Drain ``self._xye_buffer`` and write each pending XYE file.

        Drains under :attr:`_xye_lock` so workers can keep appending
        new entries while this batch's disk IO runs.  Per-file write
        errors are logged but don't abort the batch — losing one XYE
        file shouldn't kill an otherwise-valid scan.

        P3: when ``published_idxs`` is provided, only buffer entries
        whose ``img_number`` (a.k.a. ``frame.idx``) appears in the set
        are written to disk; entries for frames that finished
        integration but never got published to the .nxs are dropped.
        This keeps the XYE directory and the .nxs frame set in sync
        after a Stop mid-batch — without the filter, in-flight
        workers that the parallel dispatcher abandoned could leave
        orphan XYE files for frames that never landed in HDF5.
        """
        with self._xye_lock:
            if not self._xye_buffer:
                return
            buf = self._xye_buffer
            self._xye_buffer = []
        if published_idxs is not None:
            published_idxs = {int(i) for i in published_idxs}
            dropped = [t for t in buf if int(t[0]) not in published_idxs]
            buf = [t for t in buf if int(t[0]) in published_idxs]
            if dropped:
                logger.info(
                    'XYE: dropped %d unpublished entries (Stop mid-batch)',
                    len(dropped),
                )
        for img_number, frame in buf:
            try:
                self.save_1d(scan, frame, img_number)
            except Exception as e:
                logger.warning(
                    'XYE write failed for frame %s: %s', img_number, e,
                )

    @staticmethod
    def save_1d(scan, frame, idx):
        """Write a single-frame XYE next to the scan's .nxs file.

        Static because it only depends on the scan + frame state —
        not on per-wrangler attributes.  Filename layout matches the
        prior imageWrangler convention so existing downstream tools
        keep working: ``<scan_dir>/<scan_name>/iq_<scan>_NNNN.xye``
        (or ``itth_...`` for 2θ units).
        """
        if frame.int_1d is None:
            return
        path = os.path.dirname(scan.data_file)
        path = os.path.join(path, scan.name)
        Path(path).mkdir(parents=True, exist_ok=True)
        r1d = frame.int_1d
        # Encode the actual 1D integration axis in the prefix so the XYE reader
        # recovers the x-axis from the name.  The old `iq if q else itth` rule
        # mislabeled every non-Q axis (GI Q_ip/Q_oop/exit) as 2θ.
        from ..display_logic import xye_prefix_for_unit
        prefix = xye_prefix_for_unit(r1d.unit)
        fname = os.path.join(
            path, f'{prefix}_{scan.name}_{str(idx).zfill(4)}.xye'
        )
        write_xye(fname, r1d.radial, r1d.intensity,
                  np.sqrt(np.abs(r1d.intensity)))

    def _save_to_disk(self, scan):
        """Persist scan state to its .nxs file (intermediate save).

        Honours the h5pool pause/resume protocol so the GUI's
        h5viewer doesn't fight the writer for the file handle, and
        the per-wrangler ``file_lock`` so reads stay quiescent
        during the write.  No-op in xye_only mode (no .nxs target).
        """
        if self.xye_only:
            return
        _get_h5pool().pause(scan.data_file)
        try:
            with self.file_lock:
                scan._save_to_nexus()
        finally:
            _get_h5pool().resume(scan.data_file)
