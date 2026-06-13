# -*- coding: utf-8 -*-
"""``ScanSessionAdapter`` — xdart's bridge over a streaming ReductionSession.

Phase 4c-1: the per-frame register+submit, the pause quiesce, and the
sink flush that were inlined in ``imageWranglerThread`` move behind one
object that wraps a streaming ``ReductionSession`` + its ``QtNexusSink``.
It owns the three irreducibly-xdart concerns the headless session can't:
the LiveFrame→Frame submit, the Qt-side stop-on-write-failure translation
(never raise into the wrangler ``run()`` loop — that tears down the
QThread), and the h5pool-bracketed sink flush.

It strictly DELEGATES quiesce to ``ReductionSession.pause`` (4a) — it never
reimplements drain — and never writes the sink itself (the session's single
writer thread does), preserving the HDF5 single-writer invariant.

In Phase 4f this becomes a thin bridge over the public
``xrd_tools.session.ScanSession``; today it wraps the ReductionSession
directly so 4c is offscreen-gatable without the public API yet existing.
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
            return False
        return bool(accepted)

    # -- pause: quiesce (drain + flag) is SEPARATE from flush, so the
    #    wrangler's serial-vs-streaming routing in _enter_pause is preserved
    def quiesce(self, timeout=None) -> bool:
        """Pause the writer at a frame boundary (delegates to
        ReductionSession.pause: sets is_paused + drains).  Returns whether
        the writer fully quiesced (False = timed out / cancelled)."""
        return self._session.pause(timeout=timeout)

    def flush(self) -> None:
        """Force the sink's incremental flush (streaming pause/finish).  The
        h5pool bracket lives inside QtNexusSink._flush, on the writer
        thread — unmoved."""
        self._sink._flush(force=True)

    def resume(self) -> None:
        """Re-allow submit() after a pause (delegates to
        ReductionSession.resume).  No-op if not paused / finished."""
        self._session.resume()
