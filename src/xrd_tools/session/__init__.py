# -*- coding: utf-8 -*-
"""``xrd_tools.session`` — the headless scan-session layer (greenfield Difference 2).

A thin, Qt-free facade over a streaming :class:`~xrd_tools.reduction.ReductionSession`
+ a :class:`~xrd_tools.reduction.ReductionSink` that turns "a frame was reduced"
into an immutable :class:`FrameEvent` delivered to plain callbacks.  It is the
data-ownership boundary the roadmap calls the biggest steer: a GUI (or a
notebook, or a script) drives a scan by *commands in* and *events out*, owning no
reduction state itself.

See ADR-0003 (single-result events / multi-result records) and ADR-0004 (event
threading + generation + flush contract).
"""
from __future__ import annotations

from .scan_session import (
    FrameEvent,
    ProgressEvent,
    ScanSession,
    StateChangeEvent,
)

__all__ = [
    "ScanSession",
    "FrameEvent",
    "ProgressEvent",
    "StateChangeEvent",
]
