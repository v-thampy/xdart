"""Significant events in xdart's log (maintainer, 2026-10-24: "delineate new scans/runs with an empty line and
something like **** New Scan **** and end with a similar line"; "the date and time included at the beginning of the
line for significant events (like the start and end of scans or runs)", not for routine lines; "Add something like
==== New Run ==== to differentiate between runs and scans (which have ***). Also a blank line after the end of each
scan").

A Run opens with a dated ``==== New Run … ====`` line and closes with ``==== End of Run … ====``; each scan it
processes opens with ``**** New Scan … ****`` and closes with ``**** End of Scan … ****``. An empty line comes before
each opening banner and after each closing one (one where they meet). A significant event (a scan skipped, stopped or
reduced again; a warning or worse) shows its date and time in the log panel too, where routine lines show only the
time of day; every line of the log file is dated already.

Diagnostics (timings, resources, pipeline settings) are for diagnosis, not the regular log (maintainer, 2026-10-24):
they are logged at INFO while performance monitoring is on (Config ▸ Debug ▸ Performance Monitoring, or
``XDART_PERF``) and at DEBUG otherwise, so Config ▸ Debug ▸ Debug Logging shows them too.
"""
from __future__ import annotations

import logging
import os
import weakref

#: A record's ``extra`` key: its banner's kind, one of :data:`BANNERS`.
BANNER = "xdart_banner"
#: A record's ``extra`` key marking a significant event.
EVENT = "xdart_event"
#: Each banner kind's rule, and whether it opens its block (an empty line before) or closes it (an empty line after).
BANNERS = {"run": ("====", True), "run-end": ("====", False), "scan": ("****", True), "scan-end": ("****", False)}
_DATED = "%Y-%m-%d %H:%M:%S"


def banner(logger: logging.Logger, text: str, *args: object, kind: str = "scan") -> None:
    """``<date and time> **** text ****`` (a scan's) or ``==== text ====`` (a Run's) on its own line."""
    if kind not in BANNERS:
        raise ValueError(f"unknown banner kind {kind!r}")
    logger.info(text, *args, extra={BANNER: kind})


def event(logger: logging.Logger, level: int, text: str, *args: object) -> None:
    """A significant event: dated wherever the log is shown."""
    logger.log(level, text, *args, extra={EVENT: True})


def performance_monitoring() -> bool:
    """Whether performance monitoring is on: ``XDART_PERF`` (Config ▸ Debug ▸ Performance Monitoring sets it), or the
    within-run quartile timing of Config ▸ Performance Diagnostics…."""
    return bool(os.environ.get("XDART_PERF")) or os.environ.get("XDART_PERF_QUARTILES", "").strip() == "1"


def diagnostic(logger: logging.Logger, text: str, *args: object, shown: bool | None = None) -> None:
    """A measurement or setting for diagnosis: INFO while performance monitoring is on (``shown``: a Run's own
    reading of it, taken at its start), DEBUG otherwise."""
    on = performance_monitoring() if shown is None else shown
    logger.log(logging.INFO if on else logging.DEBUG, text, *args)


class EventFormatter(logging.Formatter):
    """``fmt`` lines, with ``datefmt``; a banner as ``<date and time> <rule> message <rule>`` alone on its line,
    with the empty lines around it; a significant event, or a warning or worse, with its full date and time where
    ``datefmt`` shows less. Each handler has its own formatter, which remembers whether its last line ended with an
    empty one."""

    def __init__(self, fmt: str, datefmt: str | None = None) -> None:
        super().__init__(fmt, datefmt)
        self._dated = None if datefmt is None else logging.Formatter(fmt, _DATED)
        self._spaced = False  # the last line ended with an empty one
        self._last: tuple[weakref.ref, str] | None = None

    def format(self, record: logging.LogRecord) -> str:
        last = self._last
        if last is not None and last[0]() is record:
            return last[1]  # the same record again: a rotating file handler formats each record twice
        text = self._line(record)
        self._last = (weakref.ref(record), text)
        return text

    def _line(self, record: logging.LogRecord) -> str:
        kind = BANNERS.get(getattr(record, BANNER, None))
        if kind is None:
            self._spaced = False
            if self._dated is not None and (getattr(record, EVENT, False) or record.levelno >= logging.WARNING):
                return self._dated.format(record)
            return super().format(record)
        rule, opens = kind
        line = f"{self.formatTime(record, _DATED)} {rule} {record.getMessage()} {rule}"
        if opens:
            line, self._spaced = (line if self._spaced else "\n" + line), False
        else:
            line, self._spaced = line + "\n", True
        return line


__all__ = ["BANNER", "BANNERS", "EVENT", "EventFormatter", "banner", "diagnostic", "event", "performance_monitoring"]
