# -*- coding: utf-8 -*-
"""``QtNexusSink`` — xdart's v2 ``.nxs`` write as a ssrl ``ReductionSink``.

The streaming :class:`xrd_tools.reduction.ReductionSession` feeds completed
``FrameReduction``s (by frame index, out-of-order ok) to :meth:`write` on its
single writer/consumer thread.  This sink hydrates the matching ``LiveFrame``
(registered by the wrangler as it submits), makes/skips the PERF-5 thumbnail,
stashes it in-memory (``add_frame``), buffers the XYE row, frees the raw
(PERF-3), and owns the mode-aware save cadence — exactly the old Phase-2 write,
relocated behind the sink interface so batch and a non-batch *reprocess* share
one write path.  NOTE: true-live *watching* (Phase 3 — the detector-rate file
watcher) intentionally keeps its own serial ``_process_one`` + direct
``_save_to_nexus`` write; it's a second, deliberate write path (one frame at a
time, parallelism moot), not a gap.

Design (per the WS-X1 Phase-2 review notes):

* **Single-writer invariant** — only the session's one writer thread calls
  begin/write/replace/finish, so all the existing thread-safety (``file_lock``,
  h5pool pause/resume) applies on that thread.  ``ssrl`` never imports this
  class; it is *passed in* (duck-typed against the ``ReductionSink`` protocol).
* **Bounded register map** — :meth:`write` POPs the ``LiveFrame`` from the
  ``{index: LiveFrame}`` map, so the map only ever holds in-flight frames, not
  all N (otherwise it would be a third reference pinning every ~1 MB cake for
  the whole scan).
* **Persist-before-evict** — the save cadence forces a flush before
  ``LiveFrameSeries._in_memory`` (cap 64) could evict an unsaved frame (the same
  invariant as the data-loss fix; ``_save_to_nexus`` calls ``mark_persisted``).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Margin below the in-memory frame cache cap at which the sink forces a save,
# so an unsaved frame is never evicted (data-loss invariant).
_SAVE_BEFORE_EVICT_MARGIN = 8


class QtNexusSink:
    """A ``ReductionSink`` that drives the existing xdart v2 writer.

    Parameters
    ----------
    host
        The ``imageWranglerThread`` (provides ``file_lock``, ``_xye_lock`` /
        ``_xye_buffer`` / ``_flush_xye_buffer``, ``LIVE_SAVE_INTERVAL``,
        ``xye_only`` / ``batch_mode`` / ``gi`` / ``incidence_motor`` /
        ``series_average``, and ``sigUpdate``).
    scan
        The ``LiveScan`` whose ``.nxs`` is being written.
    plan
        The active ``ReductionPlan`` (for the per-frame normalization factor).
    mask
        The detector-level global mask used for thumbnails.
    """

    def __init__(self, host, scan, plan, *, mask=None):
        self._host = host
        self._scan = scan
        self._plan = plan
        self._mask = mask
        self._registry: dict[int, Any] = {}
        self._since_save = 0
        self._published: set[int] = set()

    # -- registration -----------------------------------------------------
    def register(self, live_frame) -> None:
        """Hand the sink the ``LiveFrame`` for an index the wrangler is about to
        submit, so :meth:`write` can hydrate it.  Popped once written."""
        self._registry[int(live_frame.idx)] = live_frame

    # -- ReductionSink protocol -------------------------------------------
    def begin(self, scan, plan) -> None:
        self._since_save = 0
        self._published.clear()

    def write(self, frame, reduction) -> None:
        live = self._registry.pop(int(frame.index), None)
        if live is None:
            # Fail LOUD (BLOCKER-2 discipline): silently returning here made
            # the session count the frame as successfully written while its
            # data never reached the .nxs or the display.  Raising routes
            # through the writer loop's failure recording -> finish() reports
            # the run failed.
            raise RuntimeError(
                f"QtNexusSink.write: no LiveFrame registered for index "
                f"{int(frame.index)} — the wrangler must register() every "
                f"frame before submitting it"
            )
        self._hydrate(live, frame, reduction)
        self._stash_and_buffer(live)
        self._published.add(int(live.idx))
        self._publish_display(live)      # live-mode per-frame GUI hand-off (no-op in batch)
        self._emit_frame_status(live)    # status label tracks COMPLETION
        self._since_save += 1
        if self._due_to_save():
            self._flush()

    def _emit_frame_status(self, live) -> None:
        """Per-frame status at write/completion time: '<name>' (or
        '<master> #<frame>' for multi-frame sources), filename middle-
        truncated to <=30 chars.  Emitted here rather than at submit so the
        label tracks what the plots show instead of racing ahead of the
        parallel pipeline."""
        try:
            import os
            from .image_wrangler_thread import _raw_lives_in_source
            src = str(getattr(live, 'source_file', '') or '')
            trunc = getattr(self._host, '_middle_truncate', None)
            name = os.path.basename(src)
            if callable(trunc):
                name = trunc(name, max_len=30)
            if not name:
                name = f'frame {live.idx}'
            elif _raw_lives_in_source(src):
                name = f'{name} #{live.idx}'
            sig = getattr(self._host, 'showLabel', None)
            if sig is not None:
                sig.emit(name)
        except Exception:
            logger.debug("frame status emit failed", exc_info=True)

    def _publish_display(self, live) -> None:
        """Live-mode per-frame display hand-off.  Mirrors the SERIAL path
        (``image_wrangler_thread._process_one``: ``_published_frames[idx] = frame;
        sigUpdate.emit(idx)``): the writer/worker threads do ZERO Qt/display work
        — they only stash the fully-hydrated ``LiveFrame`` into the host's
        ``_published_frames`` map and emit a lightweight queued ``sigUpdate``.
        The GUI thread's ``static_scan_widget.update_data`` consumer (coalesced
        by the ~200 ms timer) then does ALL the display work: ``copy_for_display``,
        the ``data_1d`` / ``data_2d`` mirrors, ``publication_store.upsert`` (the
        cake's ONLY render source), and ``scan_data`` accumulation — going through
        the same auto-follow-vs-manual-selection arbitration as serial.

        This is the single-source-of-truth live-display contract every other live
        path already uses; doing the heavy copy/dict-build + the high-rate emit on
        the session's single WRITER thread (which also owns the .nxs flush) was
        what blanked the cake (publication never populated), stuttered the GUI
        (writer-thread Qt work flooding the coalescer), and fought the selection.
        No-op in batch (silent run; the GUI reloads from the .nxs at end-of-batch).
        """
        if getattr(self._host, "batch_mode", True):
            return
        idx = int(live.idx)
        published = getattr(self._host, "_published_frames", None)
        if published is not None:
            published[idx] = live
        sig = getattr(self._host, "sigUpdate", None)
        if sig is not None:
            try:
                sig.emit(idx)
            except Exception:
                logger.debug("sigUpdate emit failed for %s", idx, exc_info=True)

    def worker_process(self, frame, reduction) -> None:
        """Per-frame prep run on the POOL worker thread (PARALLEL), not the
        single writer thread.  The ssrl streaming worker calls this right after
        integration, so the expensive ~per-frame thumbnail is fanned out across
        the pool instead of serializing on the one writer thread (the only thing
        that made streaming 2D slower than chunked).  Order: make the PERF-5-
        gated thumbnail from the raw, then ``free_raw`` (PERF-3).  The writer
        (:meth:`write`) then ONLY stashes + writes the already-prepared frame —
        it never makes a thumbnail, preserving the single-writer invariant.
        Reads the register map WITHOUT popping (the writer pops in ``write``).
        """
        live = self._registry.get(int(frame.index))
        if live is None:
            return
        skip = (
            self._host.xye_only
            or (hasattr(live, "can_skip_thumbnail")
                and live.can_skip_thumbnail(getattr(self._scan, "skip_2d", False)))
        )
        if not skip:
            try:
                live.make_thumbnail(global_mask=self._mask)
            except Exception as e:
                logger.warning("QtNexusSink thumbnail failed for %s: %s",
                               getattr(live, "idx", "?"), e)
        # PERF-3: free the raw in BATCH mode only.  In live (non-batch) mode the
        # display needs map_raw (the 2D raw panel + the bounded data_2d window
        # read it), so keep it — live RAM is already bounded by data_2d (max 20)
        # + _in_memory (cap 64).
        if getattr(self._host, "batch_mode", True):
            live.free_raw()

    def replace(self, frame, reduction) -> None:
        # Re-fed index (reintegration): hydrate + upsert in memory, but do not
        # advance the new-frame save counter or re-buffer XYE (the original
        # write already did).  add_frame / _save_to_nexus upsert by idx (A1).
        idx = int(frame.index)
        live = self._registry.pop(idx, None)
        if live is None and idx in self._scan.frames.index:
            live = self._scan.frames[idx]
        if live is None:
            # Fail LOUD (same contract as write()): a re-fed index whose
            # LiveFrame is in neither the registry nor the scan means the
            # ORIGINAL write failed — silently returning would drop the
            # frame's data while the session counts it processed.
            raise RuntimeError(
                f"QtNexusSink.replace: no LiveFrame for re-fed index {idx} "
                f"(original write likely failed)")
        self._hydrate(live, frame, reduction)
        if not self._host.xye_only:
            self._add_frame(live)
        live.free_raw()

    def finish(self, result) -> None:
        self._flush(force=True)
        # T0-8: frames whose reduction failed or was cancelled mid-flight were
        # never popped by write()/replace() — left in the registry they pin
        # their LiveFrames (and, since batch worker_process frees raw only on
        # the success path, their full raw images) for the scan's lifetime.
        self._registry.clear()
        if getattr(self._host, "batch_mode", False):
            sig = getattr(self._host, "sigUpdate", None)
            if sig is not None:
                sig.emit(-1)

    def abort(self, result) -> None:
        # Flush whatever completed + release locks; never delete — we write into
        # the live .nxs incrementally, not a temp file.
        try:
            self._flush(force=True)
        except Exception:
            logger.exception("QtNexusSink.abort flush failed")
        self._registry.clear()     # T0-8: see finish()

    # -- internals --------------------------------------------------------
    def _hydrate(self, live, frame, reduction) -> None:
        # Cheap reference copy of the integration products onto the LiveFrame.
        # The expensive thumbnail + free_raw is done in PARALLEL by
        # ``worker_process`` (on the pool worker), not here on the writer thread.
        live.int_1d = reduction.result_1d
        live.int_2d = reduction.result_2d
        try:
            from xdart.modules.reduction import _frame_norm
            live.map_norm = _frame_norm(frame, self._plan)
        except Exception:
            logger.debug("map_norm hydrate failed for %s", getattr(live, "idx", "?"),
                         exc_info=True)

    def _add_frame(self, live) -> None:
        self._scan.add_frame(
            frame=live, calculate=False, update=True, get_sd=True, static=True,
            gi=getattr(self._host, "gi", False),
            th_mtr=getattr(self._host, "incidence_motor", None),
            series_average=getattr(self._host, "series_average", False),
            batch_save=True,
        )

    def _stash_and_buffer(self, live) -> None:
        # raw was already freed in worker_process (parallel); the writer only
        # stashes the integrated result + buffers the XYE row.
        if not self._host.xye_only:
            self._add_frame(live)        # in-memory stash (no disk I/O)
        with self._host._xye_lock:
            self._host._xye_buffer.append((live.idx, live))

    def _due_to_save(self) -> bool:
        if self._since_save <= 0:
            return False
        cap = getattr(self._scan.frames, "_in_memory_cap", 64)
        threshold = max(1, cap - _SAVE_BEFORE_EVICT_MARGIN)
        # In live (non-batch) mode the display save cadence can be tighter.
        if not getattr(self._host, "batch_mode", True):
            threshold = min(threshold, self._host.LIVE_SAVE_INTERVAL)
        return self._since_save >= threshold

    def _flush(self, *, force=False) -> None:
        if self._since_save <= 0 and not force:
            return
        if not self._host.xye_only:
            from .image_wrangler_thread import _get_h5pool
            _get_h5pool().pause(self._scan.data_file)
            try:
                with self._host.file_lock:
                    self._scan._save_to_nexus()   # also calls mark_persisted
            finally:
                _get_h5pool().resume(self._scan.data_file)
        self._host._flush_xye_buffer(
            self._scan, published_idxs=set(self._published),
        )
        self._published.clear()
        self._since_save = 0
