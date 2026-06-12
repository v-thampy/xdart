# -*- coding: utf-8 -*-
"""The ONE GUI coalescing idiom (greenfield Difference 7).

Burst of triggers → at-most-one emission, with the two semantics the
codebase actually needs made explicit instead of re-derived per site
(the debounce-vs-throttle confusion bit once in ``_absorb_chunk``):

* ``mode="throttle"`` — the FIRST trigger arms the timer; triggers while
  armed coalesce into the one pending firing.  At most one emission per
  interval, **latency bounded** by the interval even under a steady
  stream.  Use for high-rate per-frame display refresh (the caller keeps
  a "latest state" slot; the slot is drained on fire).
* ``mode="debounce"`` — every trigger RESTARTS the timer; fires once per
  burst, after the input goes quiet.  Use for bounded bursts (chunked
  loads) where one paint after the burst is wanted.  NOT for unbounded
  streams: a stream faster than the interval postpones the fire forever
  — force it with :meth:`flush` at end-of-burst.

``Coalescer`` also speaks the ``QTimer`` surface (``start``/``stop``/
``isActive``/``setInterval``) so it drops into sites (and test fakes)
that were written against a bare single-shot timer.
"""
from __future__ import annotations

from pyqtgraph import Qt

QtCore = Qt.QtCore


class Coalescer(QtCore.QObject):
    """Coalesce rapid triggers into at-most-one ``triggered`` emission."""

    triggered = QtCore.Signal()

    def __init__(self, interval_ms: int, *, mode: str = "throttle",
                 parent=None):
        super().__init__(parent)
        if mode not in ("throttle", "debounce"):
            raise ValueError(f"mode must be 'throttle' or 'debounce', "
                             f"got {mode!r}")
        self.mode = mode
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(int(interval_ms))
        self._timer.timeout.connect(self.triggered.emit)

    # -- the idiom ---------------------------------------------------------
    def trigger(self) -> None:
        """Register one input event (thread: GUI only, like QTimer)."""
        if self.mode == "debounce":
            self._timer.start()              # restart: fire after quiet
        elif not self._timer.isActive():
            self._timer.start()              # throttle: keep the pending fire

    def flush(self) -> None:
        """Fire NOW if a trigger is pending (end-of-burst forcing)."""
        if self._timer.isActive():
            self._timer.stop()
            self.triggered.emit()

    def cancel(self) -> None:
        """Drop any pending fire without emitting."""
        self._timer.stop()

    def is_pending(self) -> bool:
        return self._timer.isActive()

    # -- QTimer-compatible surface (drop-in for bare-timer sites/fakes) -----
    def start(self) -> None:
        self.trigger()

    def stop(self) -> None:
        self.cancel()

    def isActive(self) -> bool:              # noqa: N802 (Qt naming)
        return self.is_pending()

    def setInterval(self, ms: int) -> None:  # noqa: N802 (Qt naming)
        self._timer.setInterval(int(ms))

    def interval(self) -> int:
        return self._timer.interval()
