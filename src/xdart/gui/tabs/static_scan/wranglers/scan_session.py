# -*- coding: utf-8 -*-
"""``ScanSessionAdapter`` — xdart's thin bridge over the public
``xrd_tools.session.ScanSession`` (4f-bridge; see the Status block below).

Phase 4c-1: the per-frame register+submit, the pause quiesce, and the
sink flush that were inlined in ``imageWranglerThread`` move behind one
object that wraps the public ``ScanSession`` (built by ``open_live_scan_session``,
which arms the streaming ``ReductionSession`` internally) + its ``QtNexusSink``.
It owns the three irreducibly-xdart concerns the headless session can't:
the LiveFrame→Frame submit, the Qt-side stop-on-write-failure translation
(never raise into the wrangler ``run()`` loop — that tears down the
QThread), and the h5pool-bracketed sink flush (routed through
``ScanSession.flush``).

It strictly DELEGATES quiesce to ``ScanSession.pause`` (4a) — it never
reimplements drain — and never writes the sink itself (the session's single
writer thread does), preserving the HDF5 single-writer invariant.

**Status (4f-bridge, landed):** this adapter now wraps the PUBLIC
``xrd_tools.session.ScanSession`` (built by ``open_live_scan_session``), not a
raw ``ReductionSession`` — xdart is thin over the headless session.  ``submit``
→ ``ScanSession.submit`` (bool), ``quiesce`` → ``ScanSession.pause`` (drain +
flag), ``resume`` → ``ScanSession.resume``, ``is_paused``/``is_running`` are its
properties.  The GI streaming matrix + the live≡batch≡reload equivalence spine
+ byte-compat all stay green through it.

The live DISPLAY still flows GUI-side through ``QtNexusSink`` (the session
forwards every sink hook via its internal ``_EventSink``).  Routing display
through the session's ``on_frame_completed`` event channel
(QueuedConnection-marshalled, lossy subscriber) + mapping ``on_state_change`` →
``sigPaused``/``sigResuming`` is the OPTIONAL Part B, deferred behind a manual
live checkpoint; the architectural "thin over the public session" goal is met
without it.
"""
from __future__ import annotations

import logging

from xdart.modules.reduction import frame_from_live_frame

logger = logging.getLogger(__name__)


class ScanSessionAdapter:
    def __init__(self, host, scan, session, sink) -> None:
        self._host = host
        self._scan = scan
        self._session = session
        self._sink = sink

    # -- pass-throughs the wrangler / pause path read ---------------------
    @property
    def session(self):
        return self._session

    @property
    def sink(self):
        return self._sink

    @property
    def record_store(self):
        store = getattr(self._session, "record_store", None)
        if store is not None:
            return store
        return getattr(self._sink, "_record_store", None)

    @property
    def is_paused(self) -> bool:
        return self._session.is_paused

    @property
    def is_running(self) -> bool:
        return self._session.is_running

    # -- streaming write path --------------------------------------------
    def submit(self, live) -> bool:
        """Register the LiveFrame with the sink, then submit it to the session.

        Returns True when the session ACCEPTED the frame (it will be written).
        Returns False — never raising — in two cases the wrangler loop must
        treat as "stop feeding":

        * a RECORDED writer/sink failure (re-raised at ``submit()``'s fail-loud
          precheck): set the host command to 'stop' and return False.  Raising
          would escape the wrangler ``run()`` loop and tear down the QThread
          (the GIFreezeError trap);
        * the session DROPPED the frame (``submit`` returned False because it was
          cancelled / the writer died mid-wait): the frame was never registered
          in the session inventory nor counted submitted, so we just stop
          feeding.  The dangling sink registration is cleared by the session's
          ``finish()`` (``_registry.clear`` / T0-8) at end-of-run.
        """
        self._sink.register(live)
        try:
            accepted = self._session.submit(frame_from_live_frame(live))
        except BaseException as exc:
            msg = f'Save FAILED mid-run: {exc} — stopping the run.'
            logger.error(msg, exc_info=True)
            try:
                self._host.showLabel.emit(msg)
            except Exception:
                pass
            lock = getattr(self._host, "command_lock", None)
            if lock is not None:
                with lock:
                    self._host.command = 'stop'
            else:
                self._host.command = 'stop'
            self._unregister(live)
            return False
        if not accepted:
            # Dropped (cancelled / writer-dead mid-wait): the session never
            # registered or counted it, so roll back the sink registration too
            # rather than leave it pinned until finish().
            self._unregister(live)
            return False
        return True

    def _unregister(self, live) -> None:
        unreg = getattr(self._sink, "unregister", None)
        if callable(unreg):
            try:
                unreg(int(live.idx))
            except Exception:
                logger.debug("sink.unregister failed for %s",
                             getattr(live, "idx", "?"), exc_info=True)

    # -- pause: quiesce (drain + flag) is SEPARATE from flush, so the
    #    wrangler's serial-vs-streaming routing in _enter_pause is preserved
    def quiesce(self, timeout=None) -> bool:
        """Pause the writer at a frame boundary (delegates to
        ReductionSession.pause: sets is_paused + drains).  Returns whether
        the writer fully quiesced (False = timed out / cancelled)."""
        return self._session.pause(timeout=timeout)

    def flush(self) -> None:
        """Force an incremental flush through the PUBLIC ``ScanSession.flush``
        contract (codex P3): session → ``_EventSink`` → ``QtNexusSink.flush``,
        whose h5pool bracket runs on the writer thread — unmoved.  Routing
        through the session (not ``self._sink`` directly) keeps xdart thin over
        the public session API."""
        self._session.flush(force=True)

    def set_hydrator(self, hydrator) -> None:
        store = self.record_store
        if store is not None:
            store.set_hydrator(hydrator)

    def resume(self) -> None:
        """Re-allow submit() after a pause (delegates to
        ReductionSession.resume).  No-op if not paused / finished."""
        self._session.resume()
