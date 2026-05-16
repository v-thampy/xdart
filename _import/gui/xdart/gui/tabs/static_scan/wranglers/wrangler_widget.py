# -*- coding: utf-8 -*-
"""
@author: walroth
"""

# Standard library imports
import logging
import os
from contextlib import contextmanager
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
from xdart.modules.ewald import EwaldSphere
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

# Default cadence: flush sphere state to disk every N frames in batch
# / live modes.  Subclasses can override per-instance via the
# ``LIVE_SAVE_INTERVAL`` attribute if they want a different rhythm.
_LIVE_SAVE_INTERVAL = 8


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
        sphere_args: dict, used as **kwargs in sphere initialization.
            see EwaldSphere.
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
        sigUpdateData: int, signals a new arch has been added.
        sigUpdateFile: (str, str, bool, str, bool, bool), sends new scan_name, file name
            GI flag (grazing incidence), theta motor for GI, single_image and
            series_average flag to static_scan_Widget.
    """
    sigStart = Qt.QtCore.Signal()
    sigUpdateData = Qt.QtCore.Signal(int)
    # sigUpdateArch = Qt.QtCore.Signal(dict)
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
        self.sphere_args = {}

        self.command_queue = Queue()
        self.thread = wranglerThread(self.command_queue, self.sphere_args, self.fname, self.file_lock, self)
        self.thread.finished.connect(self.finished.emit)
        self.thread.started.connect(self.started.emit)
        self.thread.sigUpdate.connect(self.sigUpdateData.emit)
        # self.thread.sigUpdateArch.connect(self.sigUpdateArch.emit)
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
        self.thread = wranglerThread(self.command_queue, self.sphere_args, self.fname, self.file_lock, self)
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
        sphere_args: dict, used as **kwargs in sphere initialization.
            see EwaldSphere.
    
    methods:
        run: Called by start, main thread task.
    
    signals:
        sigUpdate: int, signals a new arch has been added.
        sigUpdateFile: (str, str, bool, str, bool, bool), sends new scan_name, file name
            GI flag (grazing incidence), theta motor for GI, single_image and
            series_average flag to static_scan_Widget.
        sigUpdateGI: bool, signals the grazing incidence condition has changed.
    """
    sigUpdate = Qt.QtCore.Signal(int)
    # sigUpdateArch = Qt.QtCore.Signal(dict)
    sigUpdateFile = Qt.QtCore.Signal(str, str, bool, str, bool, bool)
    sigUpdateGI = Qt.QtCore.Signal(bool)

    # Per-class override hook for save cadence — subclasses can set
    # this to a different integer if they want saves more/less often
    # than the default 8 frames.  Read inside _maybe_save() / the
    # subclass's dispatch loop.
    LIVE_SAVE_INTERVAL = _LIVE_SAVE_INTERVAL

    def __init__(self, command_queue, sphere_args, fname, file_lock,
                 parent=None):
        """command_queue: mp.Queue, queue for commands sent from parent
        sphere_args: dict, used as **kwargs in sphere initialization.
            see EwaldSphere.
        fname: str, path to data file.
        file_lock: mp.Condition, process safe lock for file access
        """
        super().__init__(parent)
        self.input_q = command_queue # thread queue
        self.sphere_args = sphere_args
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

        # In-memory hand-off of just-integrated arches to the main
        # thread so it doesn't have to round-trip through disk.  The
        # main thread's update_data consumes this dict.
        self._published_arches: dict = {}

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

    def run(self):
        """Main task. Subclasses (e.g. specThread) override this."""
        pass

    # ── Shared batch helpers ────────────────────────────────────────

    def _resolve_arch_mask(self, sphere, img_data):
        """Return a stable per-scan "bad pixel" mask cached on the sphere.

        Computed once from ``img_data < 0`` of the first frame seen
        by this sphere; reused for every subsequent frame.  Keeping
        the mask stable across frames is what lets pyFAI's CSR
        engine cache stay valid — a single pixel changing in the
        mask invalidates the cache and forces a ~250 ms LUT rebuild
        (observed on Eiger scans where saturation flicker shifts the
        mask CRC frame-to-frame; see session_may2026 lesson 1).

        Per-frame threshold filtering is NOT routed through this
        mask — see :meth:`_apply_threshold_inline` for that path
        (NaN-sentinel in the data, mask CRC unchanged).

        F3: callers in the parallel section should pre-warm the
        cache via :meth:`_prewarm_arch_mask` on the main thread
        BEFORE submitting work, so the cache is fully populated
        when N workers read it.  Without the prewarm, the first N
        workers all see ``None`` and race to write the same value —
        currently safe because every worker computes the SAME mask,
        but the invariant isn't enforced by the code and a future
        change (e.g. per-worker thresholding) could break it.
        """
        cached = getattr(sphere, '_cached_data_mask', None)
        if cached is None:
            try:
                cached = np.arange(img_data.size)[
                    np.asarray(img_data).flatten() < 0
                ]
            except (AttributeError, TypeError, ValueError) as e:
                logger.debug("arch-mask compute failed: %s", e)
                cached = None
            sphere._cached_data_mask = cached
        return cached

    @staticmethod
    @contextmanager
    def _borrow_fiber_integrator(sphere, fiber_pool, arch,
                                 *, angle_tol: float = 1e-4):
        """H2 fiber-integrator borrow.

        Yields a :class:`FiberIntegrator` for this arch's incidence
        angle, with the following preference order:

        1. **Borrow from pool** when the angle matches the prewarmed
           cache (within ``angle_tol`` degrees) — workers get their
           own deepcopy, no CSR-buffer races.  Most common path for
           sin²ψ / fixed-ω scans.
        2. **Build worker-local** when angle differs — slower
           ``promote("FiberIntegrator")`` call but only on
           ω-varying scans, and only for frames that drift.  No
           shared state, so thread-safe by construction.
        3. **Yield None** when GI isn't enabled at all — the arch
           integrators ignore the ``fiber_integrator`` kwarg in
           that case.

        Yielded values that came from the pool are returned to the
        pool on context exit (pool members are reused by subsequent
        frames).  Worker-local fi instances become garbage at exit.
        """
        # Local import: ssrl_xrd_tools.integrate.gid pulls pyFAI; we
        # don't want to drag that into every test that constructs a
        # wranglerThread for unrelated reasons.
        gi = bool(getattr(arch, "gi", False))
        if not gi:
            yield None
            return
        cached_angle = getattr(sphere, "_cached_fiber_integrator_angle", None)
        try:
            arch_angle = arch._get_incident_angle()
        except (AttributeError, ValueError):
            arch_angle = None
        if (fiber_pool is not None and cached_angle is not None
                and arch_angle is not None
                and abs(arch_angle - cached_angle) < angle_tol):
            with fiber_pool.borrow() as fi:
                yield fi
        else:
            from ssrl_xrd_tools.integrate.gid import create_fiber_integrator
            fi = create_fiber_integrator(
                arch._poni_from_integrator(),
                incident_angle=arch_angle if arch_angle is not None else 0.0,
                tilt_angle=arch.tilt_angle,
                sample_orientation=arch.sample_orientation,
                angle_unit="deg",
            )
            yield fi

    def _prewarm_arch_mask(self, sphere, img_data) -> None:
        """Populate ``sphere._cached_data_mask`` on the main thread.

        F3 — prevents the racy initialization that happens when N
        parallel workers all simultaneously see a ``None`` cache and
        each compute + write the same mask.  Computing on the main
        thread before submitting any worker means every worker only
        ever does a cache *read* against a stable value.

        Idempotent: a no-op when the cache is already set.  Called
        from each wrangler's run loop with the first frame's
        ``img_data`` before the parallel section.
        """
        if getattr(sphere, '_cached_data_mask', None) is not None:
            return
        self._resolve_arch_mask(sphere, img_data)

    def _apply_threshold_inline(self, img_data):
        """Pre-clamp pixels outside the threshold band to NaN.

        Returns a fresh float32 array with out-of-band pixels
        replaced by NaN.  pyFAI's CSR integrator skips NaN pixels
        automatically (no ``dummy``/``delta_dummy`` kwargs needed),
        and NaN propagates cleanly through bg subtraction and monitor
        normalization arithmetic inside ``arch.integrate_1d/2d``.

        No-op when ``self.apply_threshold`` is False — subclasses
        that don't expose a threshold UI inherit a free pass-through.
        """
        if not self.apply_threshold:
            return img_data
        img = np.asarray(img_data, dtype=np.float32, copy=True)
        bad = (img < self.threshold_min) | (img > self.threshold_max)
        img[bad] = _THRESHOLD_NAN
        return img

    def _flush_xye_buffer(self, sphere, published_idxs=None):
        """Drain ``self._xye_buffer`` and write each pending XYE file.

        Drains under :attr:`_xye_lock` so workers can keep appending
        new entries while this batch's disk IO runs.  Per-file write
        errors are logged but don't abort the batch — losing one XYE
        file shouldn't kill an otherwise-valid scan.

        P3: when ``published_idxs`` is provided, only buffer entries
        whose ``img_number`` (a.k.a. ``arch.idx``) appears in the set
        are written to disk; entries for arches that finished
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
        for img_number, arch in buf:
            try:
                self.save_1d(sphere, arch, img_number)
            except Exception as e:
                logger.warning(
                    'XYE write failed for frame %s: %s', img_number, e,
                )

    @staticmethod
    def save_1d(sphere, arch, idx):
        """Write a single-frame XYE next to the sphere's .nxs file.

        Static because it only depends on the sphere + arch state —
        not on per-wrangler attributes.  Filename layout matches the
        prior specWrangler convention so existing downstream tools
        keep working: ``<scan_dir>/<scan_name>/iq_<scan>_NNNN.xye``
        (or ``itth_...`` for 2θ units).
        """
        if arch.int_1d is None:
            return
        path = os.path.dirname(sphere.data_file)
        path = os.path.join(path, sphere.name)
        Path(path).mkdir(parents=True, exist_ok=True)
        r1d = arch.int_1d
        is_q = r1d.unit in ('q_A^-1', 'q_nm^-1')
        prefix = 'iq' if is_q else 'itth'
        fname = os.path.join(
            path, f'{prefix}_{sphere.name}_{str(idx).zfill(4)}.xye'
        )
        write_xye(fname, r1d.radial, r1d.intensity,
                  np.sqrt(np.abs(r1d.intensity)))

    def _save_to_disk(self, sphere):
        """Persist sphere state to its .nxs file (intermediate save).

        Honours the h5pool pause/resume protocol so the GUI's
        h5viewer doesn't fight the writer for the file handle, and
        the per-wrangler ``file_lock`` so reads stay quiescent
        during the write.  No-op in xye_only mode (no .nxs target).
        """
        if self.xye_only:
            return
        _get_h5pool().pause(sphere.data_file)
        try:
            with self.file_lock:
                sphere._save_to_nexus()
        finally:
            _get_h5pool().resume(sphere.data_file)

    def _parallel_integrate(self, items, integrate_fn, n_workers,
                             *, label="integration"):
        """Run ``integrate_fn`` over ``items`` in a ThreadPoolExecutor.

        Shared dispatch primitive used by both SPEC batch and NeXus
        chunked workers.  Each wrangler still owns its own per-item
        ``integrate_fn`` (signatures differ) and its own post-publish
        / save / xye logic.

        Behavior:
          * Submits one future per item up-front, then waits.
          * F2 cancel-fast: on Stop, calls
            ``pool.shutdown(wait=True, cancel_futures=True)`` so
            queued-but-not-running futures are dropped immediately.
            ``integrate_fn`` should ALSO check ``self.command``
            early so already-running workers can bail before the
            expensive 2D integration starts; pre-F2 the user could
            wait up to one full chunk after pressing Stop, because
            running workers kept going to completion.
          * Per-item exceptions are logged at error level and the
            corresponding arch is dropped from the result list.
          * Returns arches in idx-sorted order so on-disk
            frame_index stays monotonic.

        Returns a ``list[EwaldArch]`` with ``None`` entries elided.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if not items:
            return []

        completed: list = []
        pool = ThreadPoolExecutor(max_workers=max(1, int(n_workers)))
        try:
            futures = [pool.submit(integrate_fn, item) for item in items]
            for fut in as_completed(futures):
                if self.command == 'stop':
                    # Cancel everything still queued; let in-flight
                    # workers finish (Python doesn't pre-empt threads,
                    # but the integrate_fn's own stop checks will
                    # short-circuit before they hit pyFAI).
                    for f in futures:
                        f.cancel()
                    break
                try:
                    arch = fut.result()
                except Exception as e:
                    logger.error(
                        '[%s] worker raised: %s', label, e, exc_info=True,
                    )
                    continue
                if arch is None:
                    continue
                completed.append(arch)
        finally:
            # cancel_futures was added in 3.9; safe to assume.
            try:
                pool.shutdown(wait=True, cancel_futures=True)
            except TypeError:  # pragma: no cover  - older Python
                pool.shutdown(wait=True)

        completed.sort(key=lambda a: getattr(a, 'idx', 0) or 0)
        return completed
