# -*- coding: utf-8 -*-
"""``xrd_tools.session`` - the headless scan-session layer.

The public session classes are loaded lazily so importing a lightweight
submodule such as ``xrd_tools.session.readiness`` does not pull in reduction
writers or image-reader dependencies.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ScanSession",
    "FrameEvent",
    "FrameRecordStore",
    "ProgressEvent",
    "StateChangeEvent",
]

_SCAN_SESSION_EXPORTS = {
    "ScanSession",
    "FrameEvent",
    "ProgressEvent",
    "StateChangeEvent",
}


def __getattr__(name: str) -> Any:
    if name == "FrameRecordStore":
        value = getattr(import_module("xrd_tools.session.frame_record_store"), name)
    elif name in _SCAN_SESSION_EXPORTS:
        value = getattr(import_module("xrd_tools.session.scan_session"), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
