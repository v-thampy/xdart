# -*- coding: utf-8 -*-
"""The headless :class:`ScanSession` + its immutable event types.

``ScanSession`` wraps a streaming ``ReductionSession`` and a ``ReductionSink``.
The user's sink is wrapped in an internal event-emitting decorator
(:class:`_EventSink`) that forwards every hook the engine probes
(``begin``/``write``/``replace``/``finish``/``abort``/``worker_process``/
``flush``) and, after each ``write``/``replace``, fires ``on_frame_completed``
on the session's single writer thread — preserving the HDF5 single-writer
invariant (ADR-0004 §1).

Threading (ADR-0004): ``on_frame_completed`` and the completion-side
``on_progress`` fire on the WRITER thread; ``on_state_change`` and the
submit-side ``on_progress`` fire on the caller thread.  A callback that raises
is caught + logged — a listener can never kill the writer (the T0-7/S1
false-success trap).  A Qt bridge marshals ``on_frame_completed`` via a
``QueuedConnection``.

This module is Qt-free (numpy only via the result containers).  Save *cadence*
(FlushPolicy / persist-before-evict) is deliberately NOT here — it is an
xdart-adapter concern (ADR-0004 §4); the session only exposes ``flush`` as a
contract pass-through to the sink.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np

from xrd_tools.core import DEFAULT_MODE_KEY, FrameRecord, FrameView
from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
from xrd_tools.reduction import (
    Frame,
    ReductionPlan,
    ReductionResult,
    ReductionSession,
)
from .frame_record_store import FrameRecordStore

logger = logging.getLogger(__name__)


# ── immutable events ────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class FrameEvent:
    """One frame finished reducing (ADR-0003: single-result + the mode it was
    computed under).  Immutable; built from the engine's ``FrameReduction``."""

    frame_index: int
    mode_key: Any                      # GI (mode_1d, mode_2d) value tuple, or None
    result_1d: IntegrationResult1D | None
    result_2d: IntegrationResult2D | None
    metadata: Mapping[str, Any]
    generation: int                    # caller-owned stale-render stamp (ADR-0004 §2)
    timestamp: float                   # wall-clock completion (time.time())


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """Absolute (not delta) progress counts; may fire from two threads, so
    consumers treat it as idempotent."""

    submitted: int
    completed: int
    total: int | None


@dataclass(frozen=True, slots=True)
class StateChangeEvent:
    """Run-state transition (fires on the caller thread)."""

    is_running: bool
    is_paused: bool


# ── internal sink decorator ───────────────────────────────────────────────────

class _EventSink:
    """Wrap the user's sink: forward every probed hook, and after each
    ``write``/``replace`` fire the completion callback on the writer thread.

    Forwarding the *optional* hooks (``replace``/``abort``/``worker_process``/
    ``flush``) is essential — defining them unconditionally would otherwise make
    the engine treat a plain sink as replace/abort-capable, or (if omitted)
    disable the parallel ``worker_process`` thumbnail path.  Each forwards to the
    inner sink only when the inner sink actually provides it.
    """

    def __init__(self, inner, on_completed: Callable[[Frame, Any], None]) -> None:
        self._inner = inner
        self._on_completed = on_completed

    def begin(self, scan, plan) -> None:
        if self._inner is not None:
            self._inner.begin(scan, plan)

    def write(self, frame, reduction) -> None:
        if self._inner is not None:
            self._inner.write(frame, reduction)
        self._on_completed(frame, reduction)

    def replace(self, frame, reduction) -> None:
        inner_replace = getattr(self._inner, "replace", None)
        if callable(inner_replace):
            inner_replace(frame, reduction)
        elif self._inner is not None:
            # No replace hook → the engine would have called write(); match it.
            self._inner.write(frame, reduction)
        self._on_completed(frame, reduction)

    def finish(self, result) -> None:
        if self._inner is not None:
            self._inner.finish(result)

    def abort(self, result) -> None:
        inner_abort = getattr(self._inner, "abort", None)
        if callable(inner_abort):
            inner_abort(result)

    def worker_process(self, frame, reduction) -> None:
        wp = getattr(self._inner, "worker_process", None)
        if callable(wp):
            wp(frame, reduction)

    def flush(self, *, force: bool = False) -> None:
        f = getattr(self._inner, "flush", None)
        if callable(f):
            f(force=force)
            return
        # Interim: the xdart QtNexusSink still exposes the historical private
        # `_flush`; honour it until the bridge renames it (ADR-0004 §4).
        _f = getattr(self._inner, "_flush", None)
        if callable(_f):
            _f(force=force)


def _mode_key_from_plan(plan: ReductionPlan):
    """The GI sub-mode key (``(mode_1d, mode_2d)`` values) a result was computed
    under, or ``None`` for a standard scan — ADR-0003's per-completion mode tag."""
    gi = getattr(plan, "gi", None)
    if gi is None:
        return None
    m1 = getattr(gi, "mode_1d", None)
    m2 = getattr(gi, "mode_2d", None)
    return (getattr(m1, "value", m1), getattr(m2, "value", m2))


def _dimension_modes(mode_key: Any) -> tuple[str, str]:
    if isinstance(mode_key, tuple) and len(mode_key) == 2:
        m1, m2 = mode_key
        return str(m1 or DEFAULT_MODE_KEY), str(m2 or DEFAULT_MODE_KEY)
    return DEFAULT_MODE_KEY, DEFAULT_MODE_KEY


def _freeze_result_arrays(result):
    """Mark a result's ndarray fields read-only IN PLACE (zero-copy) so a
    FrameEvent listener cannot retroactively corrupt the shared, already-written
    arrays (the completion fires AFTER the sink's write, and the event holds the
    SAME ndarray objects the sink stored — a listener writing into them would
    poison persisted/cached data).  This makes the "immutable event" contract
    real without the deep-copy that would defeat retain_products=False.  Returns
    the same object.  Defensive: skips anything not a writeable ndarray."""
    if result is None:
        return result
    for attr in ("radial", "azimuthal", "intensity", "sigma"):
        arr = getattr(result, attr, None)
        if isinstance(arr, np.ndarray) and arr.flags.writeable:
            try:
                arr.flags.writeable = False
            except (ValueError, AttributeError):
                pass  # a view that doesn't own its data / can't toggle — leave it
    return result


# ── the session ───────────────────────────────────────────────────────────────

class ScanSession:
    """Drive a streaming scan reduction by commands in / events out.

    Construction arms the underlying streaming ``ReductionSession`` (its writer
    thread starts + ``sink.begin`` runs), so :meth:`start` is an idempotent
    confirmation.  Feed frames with :meth:`submit`; consume results by
    registering :meth:`on_frame_completed`.  Always :meth:`finish` (or use it as
    a context manager) to drain the writer + finalize the sink.

    ``record_store`` is optional and dormant for existing callers.  When supplied,
    completed frame records are upserted after the sink write.  Set
    ``record_store_persisted_on_write=True`` only for sinks whose write hook is
    durable enough that old heavy arrays may be evicted.
    """

    def __init__(
        self,
        plan: ReductionPlan,
        source: Any,
        sink: Any = None,
        *,
        executor: Any | None = None,
        inflight_max: int | None = None,
        gi_freeze_mode: str | None = None,
        cancel_token: Any | None = None,
        clear_frame_images: bool = False,
        record_store: FrameRecordStore | None = None,
        record_store_persisted_on_write: bool = False,
    ) -> None:
        self._lock = threading.RLock()
        self._frame_cbs: list[Callable[[FrameEvent], None]] = []
        self._progress_cbs: list[Callable[[ProgressEvent], None]] = []
        self._state_cbs: list[Callable[[StateChangeEvent], None]] = []
        self._submitted = 0
        self._completed = 0
        self._generation = 0
        self._mode_key = _mode_key_from_plan(plan)
        self._record_store = record_store
        self._record_store_persisted_on_write = bool(record_store_persisted_on_write)
        self._user_sink = sink
        event_sink = _EventSink(sink, self._on_completed)
        # Streaming + retain_products=False: per-frame results are delivered via
        # events (and persisted by a durable sink), so the session does not also
        # hoard every FrameReduction (the S2 ~14 GB-on-10k-frames trap).
        self._session = ReductionSession(
            plan,
            source,
            event_sink,
            execution="streaming",
            executor=executor,
            inflight_max=inflight_max,
            gi_freeze_mode=gi_freeze_mode,
            cancel_token=cancel_token,
            retain_products=False,
            # The writer nulls frame.image after each write so the source-array
            # reference doesn't pin ~18 MB/frame for the session's life (xdart's
            # PERF-3); a later consumer reloads via Frame.load_image.  Default
            # off — a notebook caller keeping the source frames opts in.
            clear_frame_images=clear_frame_images,
        )
        self._event_sink = event_sink

    # -- context manager ---------------------------------------------------
    def __enter__(self) -> "ScanSession":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Mirror ReductionSession.__exit__: never raise a fresh failure during an
        # exception unwind; surface the run failure only on a clean exit.
        self.finish(raise_on_failure=exc_type is None)

    # -- commands in -------------------------------------------------------
    def start(self) -> None:
        """Idempotent: the writer is armed at construction; emit the initial
        running state once."""
        self._emit_state()

    def submit(self, frame: Frame, image: np.ndarray | None = None) -> bool:
        """Feed one frame.

        Returns True when accepted, False when DROPPED (cancelled / writer-dead
        while waiting on a full in-flight window).  CALLER-CONTRACT VIOLATIONS
        RAISE rather than return False (mirroring ``ReductionSession.submit``):
        calling submit() after :meth:`finish`, or while paused, raises
        ``RuntimeError`` — these are misuse, kept loud on purpose, not a normal
        "dropped" outcome.  Advances submitted-progress only when accepted."""
        accepted = self._session.submit(frame, image)
        if accepted:
            with self._lock:
                self._submitted += 1
            self._emit_progress()
        return accepted

    def pause(self, timeout: float | None = None) -> bool:
        """Quiesce the writer at a frame boundary (delegates to
        ``ReductionSession.pause``).  Returns whether it fully drained."""
        drained = self._session.pause(timeout=timeout)
        self._emit_state()
        return drained

    def resume(self) -> None:
        self._session.resume()
        self._emit_state()

    def stop(self) -> None:
        """Cooperative cancel (sets the cancel token); the writer stops at the
        next boundary.  Call :meth:`finish` to drain + finalize."""
        self._session.cancel_token.cancel()
        self._emit_state()

    def finish(self, *, raise_on_failure: bool = True,
               join_timeout: float | None = None) -> ReductionResult:
        """Drain the writer, finalize the sink, return the result.  Idempotent:
        a second finish() returns the same result and does NOT re-emit a
        state-change event (so a bridge that tears down on the running→finished
        transition can't double-fire)."""
        was_running = self.is_running
        result = self._session.finish(
            raise_on_failure=raise_on_failure, join_timeout=join_timeout)
        if was_running:          # only on the real running -> finished transition
            self._emit_state()
        return result

    def flush(self, *, force: bool = False) -> None:
        """Contract pass-through to the sink's optional ``flush`` hook (ADR-0004
        §4).  No-op for a sink without one."""
        self._event_sink.flush(force=force)

    def set_generation(self, generation: int) -> None:
        """Set the stale-render stamp put on subsequent events (ADR-0004 §2).
        Caller-owned; the session never auto-advances it (esp. not on
        pause/resume)."""
        with self._lock:
            self._generation = int(generation)

    # -- state out ---------------------------------------------------------
    @property
    def is_running(self) -> bool:
        return self._session.is_running

    @property
    def is_paused(self) -> bool:
        return self._session.is_paused

    @property
    def frames_submitted(self) -> int:
        with self._lock:
            return self._submitted

    @property
    def frames_completed(self) -> int:
        with self._lock:
            return self._completed

    @property
    def scan(self):
        """The underlying session's scan (frame inventory / context)."""
        return self._session.scan

    # -- events out --------------------------------------------------------
    # Each registration returns an UNSUBSCRIBE callable so a notebook / the Qt
    # bridge / a remote client can detach without tearing down the session
    # (append-only listeners would otherwise leak across re-subscribes).  The
    # handle is idempotent — calling it twice is a no-op.
    def on_frame_completed(self, cb: Callable[[FrameEvent], None]) -> Callable[[], None]:
        return self._subscribe(self._frame_cbs, cb)

    def on_progress(self, cb: Callable[[ProgressEvent], None]) -> Callable[[], None]:
        return self._subscribe(self._progress_cbs, cb)

    def on_state_change(self, cb: Callable[[StateChangeEvent], None]) -> Callable[[], None]:
        return self._subscribe(self._state_cbs, cb)

    def _subscribe(self, registry: list, cb: Callable) -> Callable[[], None]:
        with self._lock:
            registry.append(cb)

        def _unsubscribe() -> None:
            with self._lock:
                try:
                    registry.remove(cb)
                except ValueError:
                    pass            # already removed / never present — idempotent
        return _unsubscribe

    # -- internals ---------------------------------------------------------
    def _on_completed(self, frame: Frame, reduction: Any) -> None:
        """Writer-thread completion hook (called by _EventSink after the sink
        write).  Builds the immutable FrameEvent + advances completion progress.
        A listener exception is caught — it must never escape the writer loop."""
        with self._lock:
            self._completed += 1
            generation = self._generation
            cbs = tuple(self._frame_cbs)
        event = FrameEvent(
            frame_index=int(getattr(reduction, "frame_index", getattr(frame, "index", -1))),
            mode_key=self._mode_key,
            # Freeze the shared result arrays read-only + the metadata into a
            # read-only view, so a listener can't retroactively corrupt the
            # already-written/cached data (the event is the bridge's sole data
            # contract — it must be tamper-evident).  Both are zero-copy.
            result_1d=_freeze_result_arrays(getattr(reduction, "result_1d", None)),
            result_2d=_freeze_result_arrays(getattr(reduction, "result_2d", None)),
            metadata=MappingProxyType(dict(getattr(reduction, "metadata", {}) or {})),
            generation=generation,
            timestamp=time.time(),
        )
        self._upsert_record_store(frame, event)
        for cb in cbs:
            try:
                cb(event)
            except Exception:
                logger.exception("ScanSession.on_frame_completed listener raised")
        self._emit_progress()

    def _upsert_record_store(self, frame: Frame, event: FrameEvent) -> None:
        if self._record_store is None:
            return
        mode_1d, mode_2d = _dimension_modes(event.mode_key)
        try:
            view = FrameView.from_results(
                label=event.frame_index,
                result_1d=event.result_1d,
                result_2d=event.result_2d,
                metadata_raw=event.metadata,
                metadata_numeric=getattr(frame, "metadata_numeric", None),
                incident_angle=getattr(getattr(frame, "geometry", None), "incident_angle", None),
                source_path=getattr(frame, "source_path", None),
                source_frame_index=getattr(frame, "source_frame_index", None),
            )
            self._record_store.upsert(
                FrameRecord.from_view(view, mode_1d=mode_1d, mode_2d=mode_2d),
                source_identity=getattr(frame, "source_identity", None),
                persisted=self._record_store_persisted_on_write,
            )
        except Exception:
            logger.exception("ScanSession record_store upsert failed")

    def _emit_progress(self) -> None:
        with self._lock:
            submitted, completed = self._submitted, self._completed
            cbs = tuple(self._progress_cbs)
        try:
            total = len(self._session.scan)
        except Exception:
            total = None
        event = ProgressEvent(submitted=submitted, completed=completed, total=total)
        for cb in cbs:
            try:
                cb(event)
            except Exception:
                logger.exception("ScanSession.on_progress listener raised")

    def _emit_state(self) -> None:
        event = StateChangeEvent(is_running=self.is_running, is_paused=self.is_paused)
        with self._lock:
            cbs = tuple(self._state_cbs)
        for cb in cbs:
            try:
                cb(event)
            except Exception:
                logger.exception("ScanSession.on_state_change listener raised")
