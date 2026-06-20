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
    apply_threshold_saturation_to_plan,
    compute_bad_pixel_mask,
    bad_pixel_counts,
)

# Qt imports
from pyqtgraph import Qt

# This module imports
from xdart.utils import catch_h5py_file as catch
from .hydrated_raw import clear_hydrated_raw




# M2: _reintegrate_frame (the module-level pickle-safe worker for the
# pre-M2 ProcessPoolExecutor reintegrate path) removed.  Architecture-v2
# routes reintegration through xrd_tools.reduction.run_reduction so
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
    writeError = Qt.QtCore.Signal(str)

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
        # Reintegration refreshes this store directly; the old data_1d/data_2d
        # dicts are no longer scan-display mirrors.
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
        # Cooperative stop for the (potentially minutes-long) batched
        # reintegrate loop; set by staticWidget.close() AND by the user Stop
        # button during a reintegrate.  Checked between batches -- a batch in
        # flight finishes (in live mode a batch is one frame, so Stop is
        # effectively per-frame).
        self.stop_requested = False
        # Live (per-frame) reintegrate: when True, _reintegrate_all runs serial
        # with a batch of 1 so each frame displays the instant it's reduced and
        # Stop aborts within one frame -- the interactive "watch + retune" path.
        # When False (default) it uses the fast batched-multicore path.  Set by
        # the GUI (bai_1d/bai_2d) from the integrator's Live toggle before start.
        self.reintegrate_live = False
        # Set False once a reintegrate is found to CHANGE the output shape
        # (npt/unit/axis), meaning a stopped partial save would be rejected by
        # the writer (mixed old+new rows).  The GUI Stop handler reads this to
        # WARN before discarding (vs silently losing the in-progress work).
        self.reintegrate_partial_savable = True
        # Per-frame pixel-rejection policy (Intensity Threshold + Mask
        # Saturated) snapshot onto the thread by the GUI (bai_1d/bai_2d) right
        # before a reintegrate starts, read fresh from the wrangler so the
        # CURRENT settings apply.  Applied to the reintegrate plan via
        # apply_threshold_saturation_to_plan; None => plan unchanged.
        self.threshold_config = None

    def _plan_for_reintegration(self, *, integrate_2d: bool):
        """Standard plan for a reintegrate, with the current Intensity-Threshold
        / Mask-Saturated policy applied.

        The plan-cache + session keys don't fingerprint the threshold fields, so
        the policy is layered on AFTER the cache .get (a fresh object id when it
        changes -> a fresh session; identity preserved when it doesn't ->
        session reuse).
        """
        # Each Reintegrate button recomputes ONLY its own dimension: Reintegrate
        # 2D does 2D-only, Reintegrate 1D does 1D-only.  Previously a 2D
        # reintegrate ALSO recomputed the 1D ("refresh the cached 1D entry"),
        # which (a) overwrote the good 1D with a wrong full-range/unmasked one
        # (the high-Q spike), and (b) forced a 1D-stack rewrite that the writer
        # rejects on a partial/stopped save (the _prepare_integrated_1d crash).
        # A 2D reintegrate has no business changing the 1D — leave it untouched;
        # the user clicks Reintegrate 1D to redo the 1D.
        integrate_1d = not integrate_2d
        plan = self._plan_cache.get(
            self.scan, integrate_1d=integrate_1d, integrate_2d=integrate_2d)
        return apply_threshold_saturation_to_plan(plan, self.threshold_config)

    @staticmethod
    def _plan_changes_output_shape(p1, p2, f0) -> bool:
        """True when reintegrating with this plan would CHANGE the stored output
        row shape (npt / 2D dims) vs the first stored frame -- in which case a
        partial (stopped) save is illegal (the writer forbids mixing fresh rows
        with stale ones) and the GUI must warn before discarding.

        Reads the stored result's ``.intensity`` array shape: ``int_1d`` /
        ``int_2d`` are ``IntegrationResult{1,2}D`` dataclasses, so ``np.shape``
        on the dataclass itself is ``()`` -- a ``[-1]`` index then raised
        ``IndexError``, the pre-check swallowed it, ``partial_savable`` stayed
        True, and the Stop "discard?" popup never showed on a shape-changing 1D
        reintegrate (the 2D ``set(())`` comparison only "worked" by accident and
        would even mis-flag an unchanged-npt 2D as unsavable)."""
        i1 = getattr(f0, "int_1d", None)
        s1 = np.shape(getattr(i1, "intensity", i1)) if i1 is not None else ()
        if p1 is not None and s1 and getattr(p1, "npt", None):
            if int(p1.npt) != int(s1[-1]):
                return True
        i2 = getattr(f0, "int_2d", None)
        s2 = set(np.shape(getattr(i2, "intensity", i2))) if i2 is not None else set()
        if p2 is not None and s2:
            plan_dims = {d for d in (getattr(p2, "npt_rad", None),
                                     getattr(p2, "npt_azim", None))
                         if d is not None}
            if plan_dims and s2 != plan_dims:
                return True
        return False

    @staticmethod
    def _frame_output_signature(frame):
        """(sig_1d, sig_2d) for a frame's stored/reduced result, where each sig
        is the writer-relevant fingerprint — unit, point count(s) and axis
        extents.  Two passes whose signatures differ cannot be mixed in one
        stack, so a partial (stopped) save of the changed pass is illegal.  A
        dim with no result is None."""
        def _ends(a):
            a = np.asarray(a)
            if a.size == 0:
                return (0, None, None)
            return (int(a.size), round(float(a[0]), 6), round(float(a[-1]), 6))
        i1 = getattr(frame, "int_1d", None)
        sig1 = None if i1 is None else (
            getattr(i1, "unit", None), _ends(getattr(i1, "radial", [])))
        i2 = getattr(frame, "int_2d", None)
        sig2 = None if i2 is None else (
            getattr(i2, "unit", None),
            _ends(getattr(i2, "radial", [])),
            _ends(getattr(i2, "azimuthal", [])))
        return (sig1, sig2)

    def _maybe_flag_unsavable(self, reduced_frame, do_2d: bool, label: str) -> None:
        """Once-per-pass: flip ``reintegrate_partial_savable`` to False when the
        freshly reduced output signature differs from the stored one (axis /
        unit / shape).  This is the authoritative check the npt-only pre-check
        can't make — it runs after the first frame is reduced, well before the
        user can Stop, so the discard popup stays reliable."""
        self._reint_sig_checked = True
        try:
            old1, old2 = getattr(self, "_reint_stored_sig", (None, None))
            new1, new2 = self._frame_output_signature(reduced_frame)
            old, new = (old2, new2) if do_2d else (old1, new1)
            if old is not None and new is not None and old != new:
                self.reintegrate_partial_savable = False
                logger.info(
                    "[REINT] %s partial-savable=False (axis/unit/shape changed "
                    "vs stored: %s -> %s)", label, old, new)
        except Exception:
            logger.debug("[REINT] post-reduce savability check failed",
                         exc_info=True)

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
                msg = f"Reintegration save FAILED — output .nxs may be incomplete: {exc}"
                logger.error("reintegration session WRITE FAILED on close: %s",
                             exc, exc_info=True)
                try:
                    self.writeError.emit(msg)
                except Exception:
                    logger.debug("reintegration writeError emit failed",
                                 exc_info=True)

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
            # Step 6: key the record under the real GI mode (canonical
            # bai_*_args values) so successive Integrate passes at different
            # modes ACCUMULATE into one record instead of colliding under
            # DEFAULT_MODE_KEY (the v2 reducer leaves frame.gi_* empty).  Non-GI
            # -> None -> DEFAULT (unchanged).  .view is unaffected.
            _is_gi = bool(getattr(self.scan, "gi", False))
            self.publication_store.upsert(
                publication_from_live_frame(
                    frame,
                    generation=self.publication_store.generation,
                    active_mode_1d=(
                        self.scan.bai_1d_args.get("gi_mode_1d", "q_total")
                        if _is_gi else None),
                    active_mode_2d=(
                        self.scan.bai_2d_args.get("gi_mode_2d", "qip_qoop")
                        if _is_gi else None),
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
        # Stamp the bad-pixel mask the live wrangler puts on every LiveFrame, so a
        # reintegrate masks exactly what a fresh integrate did.  A frame lazy-
        # loaded from the .nxs carries mask=None (the per-frame mask is not
        # persisted).  Uses the SAME compute_bad_pixel_mask + DISPLAY ceiling as
        # _resolve_frame_mask so live ≡ reintegrate on the same frame (incl. a
        # float-typed raw).  "Mask Saturated" is the AUTHORITATIVE on/off: OFF ->
        # mask=None -> nothing masked (saturated Bragg peaks KEPT, the user's
        # choice); ON -> negatives + uint32 sentinel + fraction-guarded ceiling.
        # Recomputed every pass (not cached on the frame) so a toggle change
        # between Reintegrate clicks is honoured.
        raw = getattr(frame, "map_raw", None)
        if raw is None:
            # The reduce needs the raw anyway; _lazy_load_raw is idempotent and
            # never raises.
            loader = getattr(frame, "_lazy_load_raw", None)
            if callable(loader):
                try:
                    loader()
                except Exception:
                    logger.debug("reintegrate prep: raw load failed for frame "
                                 "%s", getattr(frame, "idx", "?"), exc_info=True)
            raw = getattr(frame, "map_raw", None)
        if raw is not None:
            from .display_logic import integer_saturation_ceiling
            arr0 = np.asarray(raw)
            cfg = getattr(self, "threshold_config", None)
            mask_sat = bool(getattr(cfg, "mask_saturation", True)) if cfg is not None else True
            bad = compute_bad_pixel_mask(
                arr0, mask_saturation=mask_sat,
                saturation_ceiling=integer_saturation_ceiling(arr0))
            frame.mask = bad
            # [REINT-MASK] once-per-pass: show what the reintegrate is rejecting
            # (the user's ask: how many points are saturated before/after).
            if not getattr(self, "_reint_mask_logged", True):
                c = bad_pixel_counts(arr0)
                logger.info(
                    "[REINT-MASK] raw dtype=%s size=%s | uint32_dummy=%s "
                    "negative=%s sat_ceiling(opt-in)=%s -> masked(bad-pixel)=%s",
                    arr0.dtype, c["size"], c["uint32_dummy"], c["negative"],
                    c["saturation"], 0 if bad is None else int(np.size(bad)))
                self._reint_mask_logged = True
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
        """Refresh the publication store for one reintegrated frame.

        Wave 5: normal scan-display data no longer gets mirrored into
        ``data_1d``/``data_2d`` here.  The legacy render helpers adapt from the
        publication store, while the old dicts remain reserved for viewer-mode
        rows and transition-only fallback.
        """
        self._upsert_publication_for_frame(frame)
        self.update.emit(int(frame.idx))

    def _end_reintegrate_carryover(self) -> None:
        """Drop any publication carry-over not consumed by the reintegrate pass
        (Step 6 / codex P1) — so a stopped/failed pass can't leave stale records
        that a later scroll-back rehydration would merge.  In a ``finally``."""
        if self.publication_store is not None:
            self.publication_store.end_reintegrate()

    def bai_2d_all(self):
        """Integrates all frames 2d.  Thin wrapper over _reintegrate_all."""
        if getattr(self.scan, 'skip_2d', False):
            return
        try:
            self._reintegrate_all(do_2d=True)
        finally:
            self._end_reintegrate_carryover()

    def bai_1d_all(self):
        """Integrates all frames 1d.  Thin wrapper over _reintegrate_all."""
        try:
            self._reintegrate_all(do_2d=False)
        finally:
            self._end_reintegrate_carryover()

    def _reintegrate_all(self, *, do_2d: bool) -> None:
        """Shared GUI-button reintegration body for 1D and 2D paths.

        Architecture-v2 rewrite: switched from ``ProcessPoolExecutor`` over an
        eagerly-materialised frame list to **batched lazy iteration +
        xrd_tools.reduction.run_reduction**.

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
        # Fresh run starts un-stopped (a prior user Stop must not carry over).
        self.stop_requested = False
        # [REINT-MASK] diagnostics are logged once per pass, on the first frame
        # whose raw is in hand (see _prepare_frame_for_headless_reduction).
        self._reint_mask_logged = False
        live = bool(getattr(self, 'reintegrate_live', False))
        with self.data_lock:
            if do_2d:
                self.data_2d.clear()
                clear_hydrated_raw(self.data_2d)
            else:
                self.data_1d.clear()
        # Step 6: reset for a same-scan reintegrate.  begin_reintegrate empties
        # _items + bumps the generation exactly like clear() (so the mid-pass
        # display is unchanged: a partial Overall view blanks, a single frame
        # re-renders fresh as it is republished, and stale generation-checked
        # chunks are still rejected), but CARRIES OVER each frame's record so the
        # per-frame re-upsert below merges the recomputed GI mode into its
        # accumulated modes instead of wiping them.  The legacy dicts are still
        # cleared above so any transition fallback cannot show stale rows.
        if self.publication_store is not None:
            self.publication_store.begin_reintegrate()
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
            the dict assignment.
            """
            with self.scan.scan_lock:
                self.scan.frames[frame.idx] = frame
            if do_2d:
                # A standard 2D reintegrate also refreshes 1D so linked
                # viewers do not keep stale cached curves.
                self._publish_reintegrated_display(
                    frame,
                    include_2d=True,
                    refresh_1d=True,
                )
            else:
                self._publish_reintegrated_display(
                    frame,
                    include_2d=False,
                    refresh_1d=True,
                )
            # NB: _publish_reintegrated_display already emits ``update`` for this
            # frame (→ integrator_thread_update), so reintegration shows progress
            # per published frame.  Do NOT emit a second time here — that doubled
            # the cross-thread update pressure (the coalescer hid it, but it was
            # avoidable churn on slow machines).  The display reads the in-memory
            # publication store, not the .nxs (the persist save runs once at the
            # end), so there's no half-written read.

        label = '2D' if do_2d else '1D'
        # Live = serial, one frame at a time so each frame DISPLAYS the instant
        # it is reduced (per-frame ``update.emit`` below) and Stop aborts within
        # a single frame -- the interactive "watch + retune" path.  Batch (the
        # default) keeps the fast multicore dispatch.
        n_workers = 1 if live else max(1, min(max_cores, len(indices)))
        standard_plan = self._plan_for_reintegration(integrate_2d=do_2d)
        # [REINT] diagnostics: the plan's output shape (npt/unit) is what the
        # writer compares against the stored stack — a change here is exactly
        # what makes a partial (stopped) save illegal.  mask_sat/threshold show
        # whether pixel rejection is engaged for this pass.
        _p1 = getattr(standard_plan, "integration_1d", None)
        _p2 = getattr(standard_plan, "integration_2d", None)
        logger.info(
            "[REINT] start %s live=%s frames=%s | 1d_npt=%s unit=%s | "
            "2d_npt=(%s,%s) | mask_sat=%s threshold=[%s,%s]",
            label, live, len(indices),
            getattr(_p1, "npt", None), getattr(_p1, "unit", None),
            getattr(_p2, "npt_rad", None), getattr(_p2, "npt_azim", None),
            getattr(standard_plan, "mask_saturation", None),
            getattr(standard_plan, "threshold_min", None),
            getattr(standard_plan, "threshold_max", None))

        # Pre-flight: will this pass CHANGE the stored output shape?  If so a
        # stopped partial save can't be persisted (writer forbids mixed rows),
        # so the GUI warns before discarding.  Compare the plan's output npt to
        # the first stored frame's int_*/ shape (peeking frame[0] lazy-loads it).
        self.reintegrate_partial_savable = True
        # Stored output signature (unit + axis extents + shape) of the first
        # frame, captured BEFORE the loop reduces it in place.  The npt/dims
        # pre-check below is the fast path; the authoritative check happens once
        # after the first frame is reduced (_maybe_flag_unsavable), because a
        # reintegrate can change the AXIS or UNIT at the *same* npt (e.g. a
        # different radial range) — which the writer also rejects, but which the
        # plan's npt alone cannot predict.  That gap is why the Stop "discard?"
        # popup did not show on an axis-only change.
        self._reint_stored_sig = (None, None)
        self._reint_sig_checked = False
        try:
            _f0 = self.scan.frames[indices[0]]
            self._reint_stored_sig = self._frame_output_signature(_f0)
            if self._plan_changes_output_shape(_p1, _p2, _f0):
                self.reintegrate_partial_savable = False
                self._reint_sig_checked = True
            logger.info("[REINT] %s partial-savable=%s (shape vs stored)",
                        label, self.reintegrate_partial_savable)
        except Exception:
            logger.debug("[REINT] shape pre-check failed", exc_info=True)

        # Batched dispatch: lazy-load each batch right before
        # submitting it, publish results, then drop the batch's
        # frames so RAM stays bounded.  Live uses a batch of 1.
        _RE_BATCH = 1 if live else max(8, 32 * n_workers)

        # Indices actually reduced + published this pass.  On a user Stop we
        # KEEP these (persist below) instead of discarding — the frames done so
        # far stay reintegrated; only the not-yet-reached frames keep their prior
        # result.  (A re-run reintegrates everything again, overwriting.)
        processed_idxs: list[int] = []

        for i in range(0, len(indices), _RE_BATCH):
            if self.stop_requested:
                logger.warning(
                    "[REINT] %s reintegration STOPPED at frame %s/%s (%s done) "
                    "-- attempting to persist the done frames (see the next "
                    "[REINT] line for whether the partial save was allowed).",
                    label, i, len(indices), len(processed_idxs))
                break
            chunk_idxs = indices[i:i + _RE_BATCH]
            # LiveFrameSeries.__getitem__ does the lazy v2 load + sets
            # source refs / _source_root for the L1 raw loader.  Resilient to a
            # frame vanishing mid-pass (a concurrent scan.frames rebuild — e.g. a
            # scan started during the reintegrate): skip the missing index rather
            # than crashing the whole loop with a 'Frame not found' KeyError.
            frames = []
            for idx in chunk_idxs:
                try:
                    frames.append(self.scan.frames[idx])
                except KeyError:
                    logger.warning(
                        "%s reintegration: frame %s no longer present "
                        "(concurrent scan change); skipping.", label, idx)
            if not frames:
                continue
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
                processed_idxs.append(int(getattr(frame, "idx", -1)))
                # D1 RAM: every published frame is pinned in scan.frames
                # (unsaved -> un-evictable) until the single end-of-run save, so
                # N frames accumulate.  Shrink each frame's footprint now that its
                # results are published:
                #   * drop map_raw (~18 MB/frame) -- consumed by the reduce + the
                #     publication upsert; the replace-save doesn't rewrite the raw
                #     or thumbnail (is_replace guard), and it re-lazy-loads on
                #     demand for display.
                #   * for a 1D-only pass, drop the stale 2D slab the lazy-load
                #     pulled in (~2-8 MB/frame): it's unchanged and the save skips
                #     the 2D group (skip_2d forced below), so it's dead weight.
                frame.map_raw = None
                if not do_2d:
                    frame.int_2d = None
            # Authoritative once-per-pass savability check: compare the freshly
            # reduced output signature to the stored one.  Catches axis/unit
            # changes the plan-npt pre-check can't (so the Stop popup is reliable
            # for every reason the writer would reject a partial rewrite).
            # Advisory only (the writer is the real gate) -> never let it crash
            # the reduction: a failed heuristic just leaves partial_savable as-is.
            if reduced_frames and not getattr(self, "_reint_sig_checked", True):
                try:
                    self._maybe_flag_unsavable(reduced_frames[0], do_2d, label)
                except Exception:
                    logger.debug("[REINT] savability check skipped",
                                 exc_info=True)
                    self._reint_sig_checked = True
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
        # Persist exactly the frames that were reintegrated this pass (all of
        # them on a normal finish; the partial set on a Stop), so a stopped
        # reintegrate KEEPS the work done so far.  BUT: if this reintegrate
        # changed the output axis/unit/row shape, the writer's
        # _select_frames_to_write REJECTS a partial rewrite (it would mix fresh
        # rows with stale rows — a real corruption).  We honour that, never
        # bypass it: catch the ValueError, leave the .nxs untouched, and tell the
        # user the partial can't be persisted (finish the run, or use Batch).
        # The done frames remain in the in-memory publication store for this
        # session either way.
        _stopped = bool(self.stop_requested)
        replace_idxs = sorted({i for i in processed_idxs if i >= 0})
        if replace_idxs:
            from xdart.utils.h5pool import get_pool as _get_h5pool
            _get_h5pool().pause(self.scan.data_file)
            # D1: a 1D-only reintegrate must NOT rewrite the (unchanged) 2D group
            # -- that's what lets us drop the in-memory 2D slabs above.  Force
            # skip_2d so _prepare_integrated_2d short-circuits (disk 2D untouched);
            # restored in the finally.  (A 2D reintegrate keeps its own skip_2d.)
            _saved_skip_2d = getattr(self.scan, "skip_2d", False)
            if not do_2d:
                self.scan.skip_2d = True
            try:
                self.scan.save_to_nexus(
                    replace_frame_indices=replace_idxs,
                )
                logger.info(
                    "[REINT] %s saved %s reintegrated frame(s)%s.",
                    label, len(replace_idxs),
                    " (partial — stopped)" if _stopped else "")
                # D1: release the now-persisted frames so they don't stay pinned
                # in _in_memory after the run (no further stash fires to evict).
                try:
                    self.scan.frames.mark_persisted(replace_idxs)
                    n_evicted = self.scan.frames.evict_persisted_beyond_cap()
                    if n_evicted:
                        logger.info("[REINT] released %s frame(s) from memory "
                                    "after save.", n_evicted)
                except Exception:
                    logger.debug("[REINT] post-save eviction skipped",
                                 exc_info=True)
            except ValueError as exc:
                # Shape/axis/unit changed -> partial rewrite forbidden by the
                # writer (correctly).  Don't crash, don't loosen the check.
                logger.warning(
                    "[REINT] %s partial save NOT persisted (the reintegrate "
                    "changed the output axis/unit/npts, so the %s done frames "
                    "can't be mixed with the %s un-reintegrated ones): %s  Let "
                    "the reintegrate FINISH (or run it in Batch) to save.",
                    label, len(replace_idxs),
                    len(self.scan.frames.index) - len(replace_idxs), exc)
            finally:
                self.scan.skip_2d = _saved_skip_2d
                _get_h5pool().resume(self.scan.data_file)

    def bai_2d_SI(self):
        """Integrate the current frame, 2d
        """
        if getattr(self.scan, 'skip_2d', False):
            return
        idxs = self.frame_ids
        if 'Overall' in self.frame_ids:
            idxs = self.scan.frames.index
        # 2D reintegrate is 2D-only: the plan never recomputes the 1D, so the
        # existing int_1d is left untouched (see _plan_for_reintegration).
        plan = self._plan_for_reintegration(integrate_2d=True)
        # for idx in self.frames.keys():
        for idx in idxs:
            frame = self.scan.frames[int(idx)]

            self._reduce_reintegration_batch([frame], plan, n_workers=1)
            self._publish_reintegrated_display(
                frame,
                include_2d=True,
                refresh_1d=False,
            )

    def bai_1d_SI(self):
        """Integrate the current frame, 1d.
        """
        idxs = self.frame_ids
        if 'Overall' in self.frame_ids:
            idxs = self.scan.frames.index
        plan = self._plan_for_reintegration(integrate_2d=False)
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
            except Exception as e:
                # The loop must survive ANY task failure: this thread is
                # created once and never restarted, so a single OSError
                # (locked/corrupt/NFS file) escaping here used to kill file
                # loading for the rest of the session, silently.
                logger.error("Task %s failed: %s", method_name, e,
                             exc_info=True)
                traceback.print_exc()
            finally:
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
                # G1/T0-1: the repoint skips load_from_h5, so the wavelength
                # restored from the PREVIOUS file must not survive the switch.
                # getattr: tests drive this with duck-typed scan stubs.
                _clear_wl = getattr(self.scan, '_clear_persisted_wavelength', None)
                if callable(_clear_wl):
                    _clear_wl()
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
                # G1/T0-1: path-only repoint — drop the previous file's
                # restored wavelength (see _clear_persisted_wavelength).
                # getattr: tests drive this with duck-typed scan stubs.
                _clear_wl = getattr(self.scan, '_clear_persisted_wavelength', None)
                if callable(_clear_wl):
                    _clear_wl()
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
            # Invariant: the lazy frame series must read from the SAME file as the
            # scan.  The path-only branches above (no_nxs / live_run) repoint
            # scan.data_file but leave scan.frames as the init-time series whose
            # data_file is still the default .nxs.  So a later disk fallback — e.g.
            # a Reintegrate reading a frame that was evicted from the in-memory
            # cache — opens the wrong/missing file and FileNotFoundErrors (the live
            # reintegrate crash, 2026-06-18).  Repoint the existing series' data_file
            # in place rather than rebuilding it (which would discard the live
            # in-memory frames); a no-op for the else branch (already rebuilt to fname).
            _frames = getattr(self.scan, 'frames', None)
            if _frames is not None and hasattr(_frames, 'data_file'):
                _frames.data_file = self.fname
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
