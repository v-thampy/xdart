# -*- coding: utf-8 -*-
"""Flush cadence — the headless decision "is a save due now?" (Phase 4b).

Lift the cadence *decision*, not the cadence *action*.  The two GUI save-
cadence predicates (xdart's serial ``imageWranglerThread._save_due`` and
streaming ``QtNexusSink._due_to_save``) were near-duplicates that drifted
in their pressure input (a live ``unsaved_in_memory_count()`` query vs the
sink's local counter) and interval placement.  They collapse into ONE
pure policy here; each caller keeps passing what it actually has, and the
h5pool-bracketed ``_save_to_nexus`` flush *mechanism* stays in the GUI on
the single writer thread where the bracketing + single-writer invariants
live.

This module is pure: no Qt, no h5py, no numpy.  The persist-before-evict
coupling is honoured by feeding ``unsaved_in_memory`` at the call site; the
core never imports ``LiveFrameSeries``.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["FlushPolicy"]


@dataclass(frozen=True, slots=True)
class FlushPolicy:
    """When should the writer flush accumulated frames to disk?

    - ``interval``  — upper bound on save spacing (frames between flushes).
    - ``cap``       — the in-memory frame cache size (mirrors
      ``LiveFrameSeries._in_memory_cap``); ``cap - margin`` is the hard
      bound that forces a flush *before* the cache would have to evict an
      unsaved frame (persist-before-evict).
    - ``margin``    — how far below ``cap`` the hard bound sits.
    """

    interval: int = 8
    cap: int = 64
    margin: int = 8

    def hard_threshold(self) -> int:
        """The persist-before-evict bound: flush once this many frames are
        unsaved, so an unsaved frame never reaches the eviction cap."""
        return max(1, self.cap - self.margin)

    def should_flush(self, *, frames_since_flush: int,
                     unsaved_in_memory: int | None = None,
                     force: bool = False) -> bool:
        """True if a flush is due.

        ``frames_since_flush`` is the writer's local count since the last
        flush.  ``unsaved_in_memory`` is the live count of in-memory frames
        not yet persisted (the GUI's eviction-pressure signal); ``None``
        (the pure-headless path, and the streaming sink which tracks only
        its own counter) falls back to ``frames_since_flush`` for the
        pressure branch.

        ``force`` means **force-if-pending**, NOT "always flush": it bypasses
        the interval and pressure gates so any pending frames are saved at
        once (end-of-batch / pause / Stop), but with nothing pending
        (``frames_since_flush <= 0``) there is nothing to save and the answer
        is still ``False`` — a flush of an empty buffer is a no-op the caller
        need not perform.  The empty-buffer short-circuit therefore precedes
        the ``force`` check by design (codex P3: the name is narrower than it
        looks — this is the deliberate, test-pinned contract).
        """
        # nothing pending -> nothing to flush, even under force (see docstring)
        if frames_since_flush <= 0:
            return False
        if force:
            return True
        # upper bound on spacing
        if frames_since_flush >= self.interval:
            return True
        # persist-before-evict pressure: live unsaved count when supplied,
        # else the local counter (exactly today's two predicates)
        pressure = (frames_since_flush if unsaved_in_memory is None
                    else unsaved_in_memory)
        return pressure >= self.hard_threshold()
