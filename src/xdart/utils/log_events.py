"""Significant events in xdart's log (maintainer, 2026-10-24: "delineate new scans/runs with an empty line and
something like **** New Scan **** and end with a similar line"; "the date and time included at the beginning of the
line for significant events (like the start and end of scans or runs)", not for routine lines).

A Run, and each scan it processes, opens with an empty line and a dated ``**** New Run … ****`` or ``**** New Scan
… ****`` line, and closes with a dated ``**** End of … ****`` line; the end of a Run leaves an empty line after it. A
significant event (a scan skipped, stopped or reduced again; a warning or worse) shows its date and time in the log
panel too, where routine lines show only the time of day; every line of the log file is dated already.
"""
from __future__ import annotations

import logging

#: A record's ``extra`` key: "open" (an empty line before it), "close" (an empty line after it) or "mark" (neither).
BANNER = "xdart_banner"
#: A record's ``extra`` key marking a significant event.
EVENT = "xdart_event"
_DATED = "%Y-%m-%d %H:%M:%S"


def banner(logger: logging.Logger, text: str, *args: object, kind: str = "open") -> None:
    """``<date and time> **** text ****`` on its own line."""
    logger.info(text, *args, extra={BANNER: kind})


def event(logger: logging.Logger, level: int, text: str, *args: object) -> None:
    """A significant event: dated wherever the log is shown."""
    logger.log(level, text, *args, extra={EVENT: True})


class EventFormatter(logging.Formatter):
    """``fmt`` lines, with ``datefmt``; a banner as ``<date and time> **** message ****`` alone on its line; a
    significant event, or a warning or worse, with its full date and time where ``datefmt`` shows less."""

    def __init__(self, fmt: str, datefmt: str | None = None) -> None:
        super().__init__(fmt, datefmt)
        self._dated = None if datefmt is None else logging.Formatter(fmt, _DATED)

    def format(self, record: logging.LogRecord) -> str:
        kind = getattr(record, BANNER, None)
        if kind is not None:
            line = f"{self.formatTime(record, _DATED)} **** {record.getMessage()} ****"
            return "\n" + line if kind == "open" else line + "\n" if kind == "close" else line
        if self._dated is not None and (getattr(record, EVENT, False) or record.levelno >= logging.WARNING):
            return self._dated.format(record)
        return super().format(record)


__all__ = ["BANNER", "EVENT", "EventFormatter", "banner", "event"]
