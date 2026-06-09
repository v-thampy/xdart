# -*- coding: utf-8 -*-
"""
@author: walroth
"""

# Standard library imports
import logging
import os
from queue import Queue
from threading import Condition, RLock
import traceback
import numpy as np

logger = logging.getLogger(__name__)

from xdart.modules.reduction import (
    open_live_reduction_session,
    StandardPlanCache,
    reduce_live_frames,
)

# Qt imports
from pyqtgraph import Qt

# This module imports
from xdart.utils import catch_h5py_file as catch




# M2: _reintegrate_frame (the module-level pickle-safe worker for the
# pre-M2 ProcessPoolExecutor reintegrate path) removed.  Architecture-v2
# routes reintegration through ssrl_xrd_tools.reduction.run_reduction so
# xdart no longer owns a second per-frame integration engine here.


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
        self._reduction_session = None
        self._reduction_session_key = None

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
            finally:
                self._close_reduction_session()

    def _get_reduction_session(self, key, factory):
        if self._reduction_session is not None and self._reduction_session_key == key:
            return self._reduction_session
        self._close_reduction_session()
        self._reduction_session = factory()
        self._reduction_session_key = key
        return self._reduction_session

    def _close_reduction_session(self):
        session = self._reduction_session
        self._reduction_session = None
        self._reduction_session_key = None
        if session is not None:
            try:
                session.finish()
            except Exception as exc:
                # BLOCKER 2: finish() is fail-loud now.  Don't silently swallow a
                # reintegration write failure — log it at ERROR and record it so
                # the run can't pass as a clean success.
                self._reduction_write_error = exc
                logger.error("reintegration session WRITE FAILED on close: %s",
                             exc, exc_info=True)

    def _session_key(self, n_workers: int, plan):
        key = max(1, int(n_workers or 1))
        return (
            id(self.scan),
            str(getattr(self.scan, "name", "scan")),
            key,
            bool(getattr(self.scan, "gi", False)),
            bool(getattr(self.scan, "skip_2d", False)),
            id(plan),
        )

    def _upsert_publication_for_frame(self, frame) -> None:
        """Refresh the publication snapshot for one reintegrated frame."""
        if self.publication_store is None:
            return
        try:
            from xdart.modules.frame_publication import (
                publication_from_live_frame,
            )
            self.publication_store.upsert(
                publication_from_live_frame(
                    frame,
                    generation=self.publication_store.generation,
                )
            )
        except Exception:
            logger.debug(
                "reintegrate publication upsert failed for frame %s",
                getattr(frame, "idx", "?"), exc_info=True,
            )

    def _prepare_frame_for_headless_reduction(self, frame):
        if self.scan.static:
            frame.static = True
        if self.scan.gi:
            frame.gi = True
        if getattr(self.scan, "_cached_integrator", None) is not None:
            frame.integrator = self.scan._cached_integrator
        return frame

    def _reduce_reintegration_batch(self, frames, plan, *, n_workers: int = 1):
        frames = [
            self._prepare_frame_for_headless_reduction(frame)
            for frame in frames
        ]
        if not frames:
            return []
        is_gi = bool(getattr(self.scan, "gi", False))
        # GI reduction builds a fiber integrator from a PONI; the headless
        # session raises ``ValueError("GI reduction requires scan.poni.")`` if
        # neither the scan nor a frame carries one.  Pass the PONI the wrangler
        # stashed alongside the cached integrator, and guard the no-calibration
        # case with a clear message instead of letting an unhandled raise tear
        # down the reintegration thread.
        poni = getattr(self.scan, "_cached_poni", None)
        if is_gi and poni is None and getattr(frames[0], "poni", None) is None:
            logger.error(
                "GI reintegration needs a loaded PONI/calibration; none is "
                "available on the scan. Load the calibration and retry."
            )
            return []
        n_workers = n_workers if len(frames) > 1 else 1
        executor = n_workers if n_workers > 1 else None
        gi_freeze_mode = "scout_union" if is_gi else None
        session = self._get_reduction_session(
            self._session_key(n_workers, plan),
            lambda: open_live_reduction_session(
                frames,
                plan,
                scan_name=str(getattr(self.scan, "name", "scan")),
                global_mask=getattr(self.scan, "global_mask", None),
                integrator=getattr(self.scan, "_cached_integrator", None),
                poni=poni,
                executor=executor,
                chunk_size=max(1, min(n_workers, len(frames))),
                gi_freeze_mode=gi_freeze_mode,
            ),
        )
        return reduce_live_frames(
            frames,
            plan,
            scan_name=str(getattr(self.scan, "name", "scan")),
            global_mask=getattr(self.scan, "global_mask", None),
            integrator=getattr(self.scan, "_cached_integrator", None),
            poni=poni,
            session=session,
            chunk_size=max(1, min(n_workers, len(frames))),
            gi_freeze_mode=gi_freeze_mode,
        )

    def _publish_reintegrated_display(
        self,
        frame,
        *,
        include_2d: bool,
        refresh_1d: bool = True,
    ) -> None:
        """Refresh legacy display caches and the publication store together."""
        idx = int(frame.idx)
        with self.data_lock:
            if include_2d:
                self.data_2d[idx] = {
                    'map_raw': frame.map_raw,
                    'bg_raw': frame.bg_raw,
                    'mask': frame.mask,
                    'int_2d': frame.int_2d,
                    'gi_2d': frame.gi_2d,
                }
            if refresh_1d:
                self.data_1d[idx] = frame.copy_for_display(
                    include_2d=False,
                )
        self._upsert_publication_for_frame(frame)
        self.update.emit(idx)

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

        Architecture-v2 rewrite: switched from ``ProcessPoolExecutor`` over an
        eagerly-materialised frame list to **batched lazy iteration +
        ssrl_xrd_tools.reduction.run_reduction**.

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

        Now:
        * Iterate the index in batches of ``_RE_BATCH`` (default
          ``32 * n_workers``); each batch is lazy-loaded just before
          dispatch and goes out of scope after publish.
        * ``run_reduction`` owns worker-thread integration and private
          integrator copies — xdart only publishes results.
        * Stop is honoured between batches.
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
                # A standard 2D reintegrate also refreshes 1D so linked
                # viewers do not keep stale cached curves.
                self._publish_reintegrated_display(
                    frame,
                    include_2d=True,
                    refresh_1d=True,
                )
            else:
                self.scan._accumulate_bai_1d(frame)
                self._publish_reintegrated_display(
                    frame,
                    include_2d=False,
                    refresh_1d=True,
                )

        label = '2D' if do_2d else '1D'
        n_workers = max(1, min(max_cores, len(indices)))
        standard_plan = self._plan_cache.get(
            self.scan, integrate_2d=do_2d,
        )

        # Batched dispatch: lazy-load each batch right before
        # submitting it, publish results, then drop the batch's
        # frames so RAM stays bounded.
        _RE_BATCH = max(8, 32 * n_workers)

        for i in range(0, len(indices), _RE_BATCH):
            chunk_idxs = indices[i:i + _RE_BATCH]
            # LiveFrameSeries.__getitem__ does the lazy v2 load + sets
            # source refs / _source_root for the L1 raw loader.
            frames = [self.scan.frames[idx] for idx in chunk_idxs]
            try:
                reduced_frames = self._reduce_reintegration_batch(
                    frames,
                    standard_plan,
                    n_workers=n_workers,
                )
            except Exception as exc:
                logger.error(
                    "%s batch reintegration failed for frames %s-%s: %s; "
                    "retrying frame-by-frame",
                    label, chunk_idxs[0], chunk_idxs[-1], exc,
                    exc_info=True,
                )
                reduced_frames = []
                for frame in frames:
                    try:
                        reduced_frames.extend(
                            self._reduce_reintegration_batch(
                                [frame],
                                standard_plan,
                                n_workers=1,
                            )
                        )
                    except Exception as frame_exc:
                        logger.error(
                            "%s integration failed for frame %s: %s",
                            label, getattr(frame, "idx", None), frame_exc,
                            exc_info=True,
                        )
                        self.update.emit(getattr(frame, "idx", -1))
            for frame in reduced_frames:
                _publish(frame)
            # ``frames`` goes out of scope at the end of the iteration, so
            # the FIFO _in_memory_cap eviction can free old frames before
            # the next chunk loads.

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

            self._reduce_reintegration_batch([frame], plan, n_workers=1)
            self._publish_reintegrated_display(
                frame,
                include_2d=True,
                refresh_1d=not self.scan.gi,
            )

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

            self._reduce_reintegration_batch([frame], plan, n_workers=1)
            self._publish_reintegrated_display(
                frame,
                include_2d=False,
                refresh_1d=True,
            )

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
            if getattr(self, 'no_nxs', False):
                # Int 1D (XYE) writes only .xye files and never creates the
                # .nxs, so there is nothing to load — repoint the path/name
                # only.  Gated by an explicit flag (set per-run in
                # start_wrangler) rather than os.path.exists, so a genuinely
                # missing .nxs in normal mode still surfaces as a load error
                # instead of being silently treated as an empty XYE result.
                self.scan.data_file = self.fname
                self.scan.name = os.path.split(self.fname)[-1].split('.')[0]
            elif getattr(self, 'live_run', False):
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
