# -*- coding: utf-8 -*-
"""Silent-by-default structured logging for browse/render diagnosis."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable


def browse_debug_enabled() -> bool:
    value = os.environ.get("XDART_BROWSE_DEBUG", "")
    return value.lower() not in ("", "0", "false", "no", "off")


def sequence_summary(values, *, limit: int = 6) -> dict:
    if values is None:
        return {"count": 0, "first": None, "last": None, "sample": []}
    if isinstance(values, (str, bytes)):
        items = [values]
    else:
        try:
            items = list(values)
        except TypeError:
            items = [values]
    return {
        "count": len(items),
        "first": items[0] if items else None,
        "last": items[-1] if items else None,
        "sample": items[:limit],
    }


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        try:
            return [_jsonable(v) for v in value]
        except TypeError:
            pass
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def browse_debug_log(logger, event: str, *, level: str = "info", **fields) -> None:
    if not browse_debug_enabled():
        return
    payload = {
        "event": event,
        "t": round(time.monotonic(), 6),
    }
    payload.update({str(k): _jsonable(v) for k, v in fields.items()})
    message = "BROWSE_DEBUG " + json.dumps(
        payload, sort_keys=True, separators=(",", ":"))
    log = getattr(logger, level, None)
    if not callable(log):
        log = logger.info
    log(message)
