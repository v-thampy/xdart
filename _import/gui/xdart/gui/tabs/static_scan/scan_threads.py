# -*- coding: utf-8 -*-
"""
@author: walroth
"""

# Standard library imports
import logging
import os
from queue import Queue
from threading import Condition, RLock
# M2 dropped ProcessPoolExecutor; ThreadPoolExecutor + as_completed
# are imported locally in _reintegrate_all to keep the top-of-file
# imports tight.
import traceback
import numpy as np

logger = logging.getLogger(__name__)

from xdart.modules.reduction import (
    StandardPlanCache,
    dispatch_live_frame_reduction,
)

# Qt imports
from pyqtgraph import Qt

# This module imports
from xdart.utils import catch_h5py_file as catch




# M2: _reintegrate_frame (the module-level pickle-safe worker for the
# pre-M2 ProcessPoolExecutor reintegrate path) removed.  The new
# _reintegrate_all uses ThreadPoolExecutor + an inline closure
# instead — no pickling, no IPC, GIL released by pyFAI's Cython
# integration during the call.


class integratorThread(Qt.QtCore.QThread):
    """Thread for handling integration. Frees main gui thread from
    intensive calculations.
    
    attributes:
        frame: int, idx of frame to integrate
        lock: Condition, lock to handle access to thread attributes
        method: str, which method to call in run
        mg_1d_args, mg_2d_args: dict, arguments for multigeometry
            integration
        scan: LiveScan, object that does the integration.
    
    methods:
        bai_1d_all: Calls by frame integration 1D for all frames
        bai_1d_SI: Calls by frame integration 1D for specified frame
        bai_2d_all: Calls by frame integration 2D for all frames
        bai_2d_SI: Calls by frame integration 2D for specified frame
        load: Loads data 
        mg_1d: multigeometry 1d integration
        mg_2d: multigeometry 2d integration
        mg_setup: sets up multigeometry object
        run: main thread method.
        
    signals:
        update: empty, tells parent when new data is ready.
    """
    update = Qt.QtCore.Signal(int)

    def __init__(self, scan, frame, file_lock,
                 frames, frame_ids, data_1d, data_2d,
                 parent=None, data_lock=None, publication_store=None):
        super().__init__(parent)
        self.scan = scan
        self.frame = frame
        self.file_lock = file_lock
        self.frames = frames
        self.frame_ids = frame_ids
        self.data_1d = data_1d
        self.data_2d = data_2d
        # Shared PublicationStore (same instance as H5Viewer / displayframe).
        # Reintegration must refresh it alongside data_1d/data_2d, else the
        # cake panel (payload path is preferred) keeps showing
        # pre-reintegration pixels.
        self.publication_store = publication_store
        # Shared reentrant lock guarding data_1d / data_2d access.  Falls
        # back to a private lock when constructed without one.
        self.data_lock = data_lock if data_lock is not None else RLock()
        self.method = None
        self.lock = Condition()
        self.mg_1d_args = {}
        self.mg_2d_args = {}
        # C1: cached standard ReductionPlan per scan.
        self._plan_cache = StandardPlanCache()

    def run(self):
        """Calls self.method. Catches exception where method does
        not match any attributes.
        """
        with self.lock:
            method = getattr(self, self.method)
            try:
                method()
            except KeyError as e:
                logger.error("Method %s failed with KeyError: %s", self.method, e, exc_info=True)
                traceback.print_exc()

    def bai_2d_all(self):
        """Integrates all frames 2d.  Thin wrapper over _reintegrate_all."""
        if getattr(self.scan, 'skip_2d', False):
            return
        self._reintegrate_all(do_2d=True)

    def bai_1d_all(self):
        """Integrates all frames 1d.  Thin wrapper over _reintegrate_all."""
        self._reintegrate_all(do_2d=False)

    def _reintegrate_all(self, *, do_2d: bool) -> None:
        """Shared GUI-button reintegration body for 1D and 2D paths.

        M2 rewrite: switched from ``ProcessPoolExecutor`` over an
        eagerly-materialised frame list to **batched lazy iteration +
        ThreadPoolExecutor + IntegratorPool** — the same primitive
        the wranglers use.

        Why the change.  Pre-M2 the path was:
            all_frames = list(self.scan.frames)
            ProcessPoolExecutor(...).submit(_reintegrate_frame, frame, ...)

        For a v2 file that's:
        * ``list(self.scan.frames)`` triggers ``LiveFrameSeries.__iter__``,
          which lazy-loads every frame from disk sequentially BEFORE
          the first worker gets a task — seconds-to-tens-of-seconds of
          GUI-thread blocking before parallel work begins.
        * Each frame (with L1 lazy raw load) carries a multi-MB
          ``map_raw`` numpy array.  ProcessPoolExecutor pickles every
          one of those into a child process — gigabytes of IPC on a
          10k-frame Eiger scan.
        * Peak RAM holds the full list of N frames in the parent,
          defeating the ``_in_memory_cap=64`` eviction policy.

        After M2:
        * Iterate the index in batches of ``_RE_BATCH`` (default
          ``32 * n_workers``); each batch is lazy-loaded just before
          dispatch and goes out of scope after publish.
        * ``IntegratorPool`` borrows + worker-thread integration — no
          pickling cost, GIL released by pyFAI's Cython path.
        * Stop is honoured between batches (and inside workers,
          inherited from the wranglers' pattern).
        """
        with self.data_lock:
            if do_2d:
                self.data_2d.clear()
            else:
                self.data_1d.clear()
        # Drop stale publications and bump the store generation so any
        # in-flight generation-checked subscribers reject pre-reintegration
        # chunks.  Every frame is republished below via ``_publish``.
        if self.publication_store is not None:
            self.publication_store.clear()
        with self.scan.scan_lock:
            if do_2d:
                self.scan.bai_2d = None
            else:
                self.scan.bai_1d = None

        max_cores = getattr(self.scan, 'max_cores', 1)
        indices = list(self.scan.frames.index)
        if not indices:
            return

        def _publish(frame):
            """Reattach frame into scan and viewer dicts.

            N3: ``scan.frames[frame.idx] = frame`` is a scan-state
            mutation that other threads (the wrangler thread, the
            GUI's LiveFrameSeries.__getitem__) can race against.  Hold
            ``scan_lock`` while we do it.  The lock is short — just
            the dict assignment + the bai accumulator — and the
            accumulator path itself already takes scan_lock
            internally, so we don't deadlock by nesting (Condition
            is reentrant).
            """
            with self.scan.scan_lock:
                self.scan.frames[frame.idx] = frame
            if do_2d:
                self.scan._accumulate_bai_2d(frame)
                with self.data_lock:
                    self.data_2d[int(frame.idx)] = {
                        'map_raw': frame.map_raw,
                        'bg_raw': frame.bg_raw,
                        'mask': frame.mask,
                        'int_2d': frame.int_2d,
                        'gi_2d': frame.gi_2d,
                    }
                    # A standard 2D reintegrate also refreshes 1D so
                    # linked viewers do not keep stale cached curves.
                    self.data_1d[int(frame.idx)] = frame.copy_for_display(
                        include_2d=False,
                    )
            else:
                self.scan._accumulate_bai_1d(frame)
                with self.data_lock:
                    self.data_1d[int(frame.idx)] = frame.copy_for_display(
                        include_2d=False,
                    )
            # Republish the reintegrated frame so the store (the preferred
            # display source) carries the NEW pixels, not the pre-reintegrate
            # ones.  Built from the just-integrated frame, so its int_1d/int_2d
            # are current.
            if self.publication_store is not None:
                try:
                    from xdart.modules.frame_publication import (
                        publication_from_live_frame,
                    )
                    self.publication_store.upsert(
                        publication_from_live_frame(
                            frame, generation=self.publication_store.generation,
                        )
                    )
                except Exception:
                    logger.debug(
                        "reintegrate publication upsert failed for frame %s",
                        getattr(frame, "idx", "?"), exc_info=True,
                    )
            self.update.emit(frame.idx)

        label = '2D' if do_2d else '1D'
        n_workers = max(1, min(max_cores, len(indices)))
        standard_plan = self._plan_cache.get(
            self.scan, integrate_2d=do_2d,
        )

        # IntegratorPool: one deep-copied pyFAI integrator per worker.
        # If scan._cached_integrator is None (scan fresh-from-load
        # without a wrangler having attached an integrator), the pool
        # comes back None and we fall back to the serial path.
        from xdart.utils.integrator_pool import ensure_integrator_pool
        from concurrent.futures import ThreadPoolExecutor

        integrator_pool = ensure_integrator_pool(
            self.scan, '_cached_integrator', n_workers,
        )

        # Same per-worker pattern for the GI fiber integrator (H2).
        # Only relevant when scan.gi is set AND a fiber integrator
        # has been pre-built; otherwise None and the integrate calls
        # treat the fiber arg as a no-op.
        fiber_pool = None
        if (self.scan.gi
                and getattr(self.scan, '_cached_fiber_integrator', None)
                is not None):
            fiber_pool = ensure_integrator_pool(
                self.scan, '_cached_fiber_integrator', n_workers,
                pool_attr='_cached_fiber_integrator_pool',
            )

        # P2: re-use the wrangler base class's angle-aware borrow.
        # The plain ``fiber_pool.borrow()`` below was unconditionally
        # handing out the prewarmed (frame-0) FiberIntegrator to every
        # worker.  For ω-varying GI scans (e.g. sin²ψ sweeps) the
        # per-frame incidence angle drifts and the prewarmed instance
        # silently integrates every frame at frame-0 geometry —
        # silently wrong.  The helper falls back to a worker-local
        # fiber integrator built at the right angle when the
        # per-frame angle differs from ``_cached_fiber_integrator_angle``.
        from xdart.gui.tabs.static_scan.wranglers.wrangler_widget import (
            wranglerThread,
        )
        _borrow_fi = wranglerThread._borrow_fiber_integrator

        def _worker(frame):
            """Re-integrate one frame on a thread.  Borrows a private
            integrator from the pool to avoid pyFAI's CSR scratch
            buffer races; same fix as IntegratorPool in the wranglers.
            """
            if self.scan.static:
                frame.static = True
            if self.scan.gi:
                frame.gi = True
            if integrator_pool is not None:
                with integrator_pool.borrow() as ai:
                    frame.integrator = ai

                    def _legacy_gi_for_frame() -> None:
                        # P2: angle-aware fiber borrow — pool hit when the
                        # frame's incidence angle matches the cached
                        # prewarm angle (most scans), worker-local build
                        # otherwise.
                        with _borrow_fi(self.scan, fiber_pool, frame) as fi:
                            frame.integrate_1d(
                                fiber_integrator=fi,
                                **self.scan.bai_1d_args,
                            )
                            if do_2d:
                                frame.integrate_2d(
                                    fiber_integrator=fi,
                                    **self.scan.bai_2d_args,
                                )

                    dispatch_live_frame_reduction(
                        frame, self.scan,
                        standard_plan=standard_plan,
                        integrator=ai,
                        global_mask=self.scan.global_mask,
                        legacy_gi=_legacy_gi_for_frame,
                    )
                    # Detach pool integrator before the next worker
                    # borrows the same instance.
                    frame.integrator = self.scan._cached_integrator
            else:
                # Fallback: no integrator pool — still go through the
                # shared dispatch helper so the GI vs standard logic
                # stays in one place.
                def _legacy_gi_serial() -> None:
                    if do_2d:
                        frame.integrate_2d(**self.scan.bai_2d_args)
                    else:
                        frame.integrate_1d(**self.scan.bai_1d_args)

                dispatch_live_frame_reduction(
                    frame, self.scan,
                    standard_plan=standard_plan,
                    integrator=frame.integrator,
                    global_mask=self.scan.global_mask,
                    legacy_gi=_legacy_gi_serial,
                )
            return frame

        # Batched dispatch: lazy-load each batch right before
        # submitting it, publish results, then drop the batch's
        # frames so RAM stays bounded.
        _RE_BATCH = max(8, 32 * n_workers)

        if max_cores > 1 and len(indices) > 1 and integrator_pool is not None:
            for i in range(0, len(indices), _RE_BATCH):
                chunk_idxs = indices[i:i + _RE_BATCH]
                # LiveFrameSeries.__getitem__ does the lazy v2 load + sets
                # source refs / _source_root for the L1 raw loader.
                frames = [self.scan.frames[idx] for idx in chunk_idxs]
                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    futures = {
                        pool.submit(_worker, frame): frame.idx
                        for frame in frames
                    }
                    from concurrent.futures import as_completed
                    for fut in as_completed(futures):
                        try:
                            _publish(fut.result())
                        except Exception as e:
                            frame_idx = futures[fut]
                            logger.error(
                                "%s integration failed for frame %s: %s",
                                label, frame_idx, e, exc_info=True,
                            )
                            self.update.emit(frame_idx)
                # ``frames`` goes out of scope at the end of the
                # iteration, so the FIFO _in_memory_cap eviction
                # can free those frames before the next chunk loads.
        else:
            # Serial fallback (max_cores=1, single frame, or no
            # integrator pool available).  Still lazy-loaded one
            # at a time so we don't materialise the full list.
            for idx in indices:
                frame = self.scan.frames[idx]
                _publish(_worker(frame))

        # Persist recomputed int_* rows back to disk via the v2
        # replace-frames path.  The save re-writes /entry/reduction
        # so the persisted bai_*_args reflect this reintegration's
        # parameters (which is the whole reason the user kicked it
        # off).  Replaces the legacy ``ut.dict_to_h5(...,
        # 'bai_*_args')`` write-to-root path (v1 layout, dropped
        # in 0.37.0) — that path never updated the v2 stacked rows.
        #
        # K2: bracket the save with the H5FilePool pause/resume
        # protocol so any concurrent GUI h5viewer reads through the
        # pool drop their cached handles and wait for the writer to
        # release.  Wrangler save paths already do this; the
        # reintegrate path was the one save site that didn't, which
        # could race a viewer's open handle on the same .nxs file.
        replace_idxs = list(self.scan.frames.index)
        if replace_idxs:
            from xdart.utils.h5pool import get_pool as _get_h5pool
            _get_h5pool().pause(self.scan.data_file)
            try:
                self.scan.save_to_nexus(
                    replace_frame_indices=replace_idxs,
                )
            finally:
                _get_h5pool().resume(self.scan.data_file)

    def bai_2d_SI(self):
        """Integrate the current frame, 2d
        """
        if getattr(self.scan, 'skip_2d', False):
            return
        idxs = self.frame_ids
        if 'Overall' in self.frame_ids:
            idxs = self.scan.frames.index
        # C1: cached plan covers integrate_1d + integrate_2d together
        # since a 2D reintegrate also refreshes the cached 1D entry.
        plan = self._plan_cache.get(self.scan, integrate_2d=True)
        # for idx in self.frames.keys():
        for idx in idxs:
            frame = self.scan.frames[int(idx)]

            def _legacy_gi_2d(frame=frame) -> None:
                frame.integrate_2d(**self.scan.bai_2d_args)

            dispatch_live_frame_reduction(
                frame, self.scan,
                standard_plan=plan,
                integrator=frame.integrator,
                global_mask=self.scan.global_mask,
                legacy_gi=_legacy_gi_2d,
            )
            with self.data_lock:
                self.data_2d[int(idx)] = {
                    'map_raw': frame.map_raw,
                    'bg_raw': frame.bg_raw,
                    'mask': frame.mask,
                    'int_2d': frame.int_2d,
                    'gi_2d': frame.gi_2d}
                if not self.scan.gi:
                    self.data_1d[int(frame.idx)] = frame.copy_for_display(
                        include_2d=False,
                    )
            self.update.emit(idx)

    def bai_1d_SI(self):
        """Integrate the current frame, 1d.
        """
        idxs = self.frame_ids
        if 'Overall' in self.frame_ids:
            idxs = self.scan.frames.index
        plan = self._plan_cache.get(self.scan, integrate_2d=False)
        # for (idx, frame) in self.frames.items():
        for idx in idxs:
            frame = self.scan.frames[int(idx)]

            def _legacy_gi_1d(frame=frame) -> None:
                frame.integrate_1d(**self.scan.bai_1d_args)

            dispatch_live_frame_reduction(
                frame, self.scan,
                standard_plan=plan,
                integrator=frame.integrator,
                global_mask=self.scan.global_mask,
                legacy_gi=_legacy_gi_1d,
            )
            with self.data_lock:
                self.data_1d[int(frame.idx)] = frame.copy_for_display(
                    include_2d=False,
                )
            self.update.emit(frame.idx)

    def load(self):
        """Load data.
        """
        self.scan.load_from_h5()


class fileHandlerThread(Qt.QtCore.QThread):
    """Thread class for loading data. Handles locks and waiting for
    locks to be released.
    """
    sigNewFile = Qt.QtCore.Signal(str)
    sigUpdate = Qt.QtCore.Signal()
    sigTaskStarted = Qt.QtCore.Signal()
    sigTaskDone = Qt.QtCore.Signal(str)
    
    def __init__(self, scan, frame, file_lock,
                 parent=None, frame_ids=None, frames=None,
                 data_1d=None, data_2d=None, data_lock=None):
        """
        Parameters
        ----------
        file_lock : multiprocessing.Condition
        frame : xdart.modules.live.LiveFrame
        scan : xdart.modules.live.LiveScan
        data_lock : threading.RLock, optional
            Shared lock guarding data_1d / data_2d; a private RLock is
            created when not provided.

        H3: ``frame_ids``, ``data_1d``, ``data_2d`` default to None
        (was ``[]`` / ``{}`` — mutable defaults shared across all
        instances that omit the kwarg).
        """
        super().__init__(parent)
        self.scan = scan
        self.frame = frame
        self.frame_ids = frame_ids if frame_ids is not None else []
        self.frames = frames
        self.data_1d = data_1d if data_1d is not None else {}
        self.data_2d = data_2d if data_2d is not None else {}
        self.data_lock = data_lock if data_lock is not None else RLock()
        self.file_lock = file_lock
        self.queue = Queue()
        self.fname = scan.data_file
        self.new_fname = None
        self.lock = Condition()
        self.running = False
        self.update_2d = True
        # When True, ``set_datafile`` only repoints ``data_file`` at the
        # new scan instead of reloading the (lagging) on-disk frames.
        # Set by static_scan_widget for the duration of a live, non-batch
        # wrangler run — during which the GUI scan is driven entirely
        # by the in-memory per-frame hand-off and a disk reload would
        # blank the live display.  See static_scan_widget.start_wrangler.
        self.live_run = False

    def run(self):
        while True:
            method_name = self.queue.get()
            if method_name is None:
                break  # Sentinel: cleanly exit the thread
            try:
                self.running = True
                self.sigTaskStarted.emit()
                method = getattr(self, method_name)
                method()
            except KeyError as e:
                logger.error("Task %s failed with KeyError: %s", method_name, e, exc_info=True)
                traceback.print_exc()
            self.running = False
            self.sigTaskDone.emit(method_name)
    
    def set_datafile(self):
        with self.file_lock:
            skip_2d = getattr(self.scan, 'skip_2d', False)
            if getattr(self, 'live_run', False):
                # Live, non-batch run: the wrangler owns this file and
                # is feeding the GUI in-memory frames per frame.  A full
                # ``scan.set_datafile`` would call ``load_from_h5``,
                # which replaces ``scan.frames`` with a disk-backed
                # series whose index only reflects flushed frames (saves
                # are batched every LIVE_SAVE_INTERVAL).  That discards
                # the just-appended in-memory frame indices and blanks
                # the display until the next disk flush — the multi-scan
                # Eiger "plots never update" bug.  Repoint the path only;
                # new_scan() already reset the index for this scan.
                self.scan.data_file = self.fname
                self.scan.name = os.path.split(self.fname)[-1].split('.')[0]
            else:
                # O7: dropped legacy ``save_args={'compression': None}``
                # passthrough — the v2 writer (save_to_nexus) doesn't
                # accept a ``compression`` kwarg.  N5 made set_datafile's
                # defaults None-sentinels, so omitting save_args is the
                # right call.  The stale dict was stripped inside
                # set_datafile via ``save_args.pop('compression', None)``
                # but that workaround is unnecessary now that the caller
                # doesn't supply the dead kwarg in the first place.
                self.scan.set_datafile(self.fname)
            self.scan.skip_2d = skip_2d  # preserve checkbox state across load
        self.sigNewFile.emit(self.fname)
        self.sigUpdate.emit()
    
    def update_scan(self):
        with self.file_lock:
            try:
                self.scan.load_from_h5(replace=False, data_only=True,
                                         set_mg=False)
            except KeyError as e:
                logger.debug("Failed to load scan data from HDF5: %s", e)

    def load_frame(self):
        """Load a single frame via the v2 lazy loader (LiveFrameSeries.__getitem__)."""
        try:
            self.frame = self.scan.frames[self.frame.idx]
        except KeyError as e:
            logger.debug("load_frame: %s", e)
        self.sigUpdate.emit()

    def load_frames(self):
        """Populate data_1d/data_2d caches by lazy-loading frames via v2.

        LiveFrameSeries.__getitem__ now reads from the stacked
        ``entry/integrated_1d`` / ``integrated_2d`` arrays and the
        per-frame ``frames/frame_NNNN/thumbnail`` group.  No v1 frame
        groups touched.
        """
        for idx in self.frame_ids:
            try:
                frame = self.scan.frames[int(idx)]
            except (KeyError, IndexError) as e:
                logger.debug("Data missing for frame %s: %s", idx, e)
                continue
            with self.data_lock:
                self.data_1d[int(idx)] = frame.copy_for_display(include_2d=False)
                if self.update_2d:
                    self.data_2d[int(idx)] = {
                        'map_raw': getattr(frame, 'map_raw', None),
                        'bg_raw': getattr(frame, 'bg_raw', 0),
                        'mask': getattr(frame, 'mask', None),
                        'int_2d': getattr(frame, 'int_2d', None),
                        'gi_2d': getattr(frame, 'gi_2d', {}),
                        'thumbnail': getattr(frame, 'thumbnail', None),
                    }
        self.sigUpdate.emit()

    def save_data_as(self):
        if self.new_fname is not None and self.new_fname != "":
            with self.file_lock:
                with catch(self.scan.data_file, 'r') as f1:
                    with catch(self.new_fname, 'w') as f2:
                        for key in f1:
                            f1.copy(key, f2)
                        for attr in f1.attrs:
                            f2.attrs[attr] = f1.attrs[attr]
        self.new_fname = None
