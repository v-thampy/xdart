"""The log's Run and scan banners and its dated significant events (maintainer, 2026-10-24: "delineate new
scans/runs with an empty line and something like **** New Scan **** and end with a similar line"; "the date and time
included at the beginning of the line for significant events (like the start and end of scans or runs)")."""
from __future__ import annotations

import logging
import time

from xdart.gui.widgets.log_panel import LOG_FORMAT, LogFeed
from xdart.utils import log_events
from xdart.utils.log_events import EventFormatter

_AT = time.mktime((2026, 10, 24, 15, 40, 12, 0, 0, -1))


def _record(message, *args, level=logging.INFO, **extra):
    record = logging.LogRecord("xdart.test", level, __file__, 1, message, args, None)
    record.created, record.msecs = _AT, 0.0
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_a_banner_stands_alone_on_a_dated_line_and_opens_or_closes_with_an_empty_line():
    panel, file = EventFormatter(LOG_FORMAT, "%H:%M:%S"), EventFormatter(LOG_FORMAT)
    for formatter in (panel, file):
        assert formatter.format(_record("New Run · %s", "Int 1D", **{log_events.BANNER: "open"})) == (
            "\n2026-10-24 15:40:12 **** New Run · Int 1D ****")
        assert formatter.format(_record("End of Scan · a · 3 frames · 2.0 s", **{log_events.BANNER: "mark"})) == (
            "2026-10-24 15:40:12 **** End of Scan · a · 3 frames · 2.0 s ****")
        assert formatter.format(_record("End of Run", **{log_events.BANNER: "close"})) == (
            "2026-10-24 15:40:12 **** End of Run ****\n")


def test_the_panel_dates_significant_events_and_warnings_but_not_routine_lines():
    panel = EventFormatter(LOG_FORMAT, "%H:%M:%S")
    assert panel.format(_record("scrolled")) == "15:40:12 INFO xdart.test: scrolled"
    assert panel.format(_record("rebuilt", **{log_events.EVENT: True})) == (
        "2026-10-24 15:40:12 INFO xdart.test: rebuilt")
    assert panel.format(_record("refused", level=logging.WARNING)) == "2026-10-24 15:40:12 WARNING xdart.test: refused"
    # every line of the log file is dated already
    assert EventFormatter(LOG_FORMAT).format(_record("scrolled")).startswith("2026-10-24 15:40:12,000 INFO")


def test_the_log_panel_feed_shows_the_banner_after_an_empty_line():
    feed = LogFeed(capacity=10)
    logger = logging.getLogger("xdart.test.feed")
    logger.addHandler(feed)
    logger.setLevel(logging.INFO)
    try:
        log_events.banner(logger, "New Scan · %s", "scanb_int1d.nexus")
        logger.info("routine")
    finally:
        logger.removeHandler(feed)
    lines, _end = feed.held()
    assert lines[0].startswith("\n") and lines[0].endswith(" **** New Scan · scanb_int1d.nexus ****")
    assert lines[0][1:5].isdigit() and lines[1].endswith("INFO xdart.test.feed: routine")
