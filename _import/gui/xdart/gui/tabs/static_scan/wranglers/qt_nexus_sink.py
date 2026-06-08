# -*- coding: utf-8 -*-
"""``QtNexusSink`` — xdart's v2 ``.nxs`` write as a ssrl ``ReductionSink``.

The streaming :class:`ssrl_xrd_tools.reduction.ReductionSession` feeds completed
``FrameReduction``s (by frame index, out-of-order ok) to :meth:`write` on its
single writer/consumer thread.  This sink hydrates the matching ``LiveFrame``
(registered by the wrangler as it submits), makes/skips the PERF-5 thumbnail,
stashes it in-memory (``add_frame``), buffers the XYE row, frees the raw
(PERF-3), and owns the mode-aware save cadence — exactly the old Phase-2 write,
relocated behind the sink interface so live and batch can share one write path.

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
            logger.error(
                "QtNexusSink.write: no LiveFrame registered for index %s",
                int(frame.index),
            )
            return
        self._hydrate(live, frame, reduction)
        self._publish_display(live)      # live-mode per-frame GUI publish (no-op in batch)
        self._stash_and_buffer(live)
        self._published.add(int(live.idx))
        self._since_save += 1
        if self._due_to_save():
            self._flush()

    def _publish_display(self, live) -> None:
        """Live-mode per-frame display publish: populate the GUI's data_1d /
        data_2d caches and emit the per-frame sigUpdate so the view refreshes as
        each frame lands.  Runs on the writer thread (the same background-thread
        contract the old _process_one used).  No-op in batch mode (silent run).
        The raw was NOT freed for live (worker_process gates free_raw on batch),
        so map_raw is present for the 2D panel.
        """
        if getattr(self._host, "batch_mode", True):
            return
        idx = int(live.idx)
        try:
            self._host.data_1d[idx] = live.copy_for_display(include_2d=False)
            self._host.data_2d[idx] = {
                'map_raw': live.map_raw, 'bg_raw': live.bg_raw,
                'mask': live.mask, 'int_2d': live.int_2d,
                'gi_2d': live.gi_2d, 'thumbnail': None,
            }
        except Exception:
            logger.debug("live display publish failed for %s", idx, exc_info=True)
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
            return
        self._hydrate(live, frame, reduction)
        if not self._host.xye_only:
            self._add_frame(live)
        live.free_raw()

    def finish(self, result) -> None:
        self._flush(force=True)
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
