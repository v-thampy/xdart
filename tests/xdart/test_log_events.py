"""The log's Run and scan banners and its dated significant events (maintainer, 2026-10-24: "delineate new
scans/runs with an empty line and something like **** New Scan **** and end with a similar line"; "the date and time
included at the beginning of the line for significant events (like the start and end of scans or runs)"; "Add
something like ==== New Run ==== to differentiate between runs and scans (which have ***). Also a blank line after the
end of each scan"), and diagnostics kept out of the regular log."""
from __future__ import annotations

import logging
import time
from logging.handlers import RotatingFileHandler

import pytest

from xdart.gui.widgets.log_panel import LOG_FORMAT, LogFeed
from xdart.utils import log_events
from xdart.utils.log_events import EventFormatter

_AT = time.mktime((2026, 10, 24, 15, 40, 12, 0, 0, -1))
_DATE = "2026-10-24 15:40:12"


def _record(message, *args, level=logging.INFO, **extra):
    record = logging.LogRecord("xdart.test", level, __file__, 1, message, args, None)
    record.created, record.msecs = _AT, 0.0
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def _banner(kind, message):
    return _record(message, **{log_events.BANNER: kind})


@pytest.mark.parametrize("datefmt", ["%H:%M:%S", None], ids=["panel", "file"])
def test_runs_and_scans_have_their_own_rules_and_one_empty_line_between_blocks(datefmt):
    formatter = EventFormatter(LOG_FORMAT, datefmt)
    lines = [formatter.format(record) for record in (
        _banner("run", "New Run · Int 1D"), _banner("scan", "New Scan · a"), _record("routine"),
        _banner("scan-end", "End of Scan · a · 1 frame · 2.0 s"), _banner("scan", "New Scan · b"),
        _banner("scan-end", "End of Scan · b · 3 frames · 1.0 s"), _banner("run-end", "End of Run · Finished"),
        _banner("run", "New Run · Int 2D"))]
    assert lines[0] == f"\n{_DATE} ==== New Run · Int 1D ===="
    assert lines[1] == f"\n{_DATE} **** New Scan · a ****"
    assert lines[3] == f"{_DATE} **** End of Scan · a · 1 frame · 2.0 s ****\n"
    assert lines[4] == f"{_DATE} **** New Scan · b ****"  # the scan's end left the empty line
    assert lines[5] == f"{_DATE} **** End of Scan · b · 3 frames · 1.0 s ****\n"
    assert lines[6] == f"{_DATE} ==== End of Run · Finished ====\n"
    assert lines[7] == f"{_DATE} ==== New Run · Int 2D ===="
    assert "\n\n\n" not in "\n".join(lines)  # one empty line where blocks meet, never two
    with pytest.raises(ValueError):
        log_events.banner(logging.getLogger("xdart.test"), "x", kind="mark")


def test_a_rotating_log_file_gets_each_banner_once_with_its_empty_line(tmp_path):
    """A rotating file handler formats each record twice (its size check, then its write)."""
    path = tmp_path / "xdart.log"
    handler = RotatingFileHandler(path, maxBytes=1 << 20, backupCount=1, encoding="utf-8")
    handler.setFormatter(EventFormatter(LOG_FORMAT))
    logger = logging.getLogger("xdart.test.rotating")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        logger.info("routine")
        log_events.banner(logger, "End of Scan · a · 2 frames · 1.0 s", kind="scan-end")
        log_events.banner(logger, "New Scan · b", kind="scan")
        log_events.banner(logger, "End of Run · Finished", kind="run-end")
    finally:
        logger.removeHandler(handler)
        logger.propagate = True
        handler.close()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [line.split(" ", 2)[2] for line in lines if line] == [
        "INFO xdart.test.rotating: routine", "**** End of Scan · a · 2 frames · 1.0 s ****", "**** New Scan · b ****",
        "==== End of Run · Finished ===="]
    assert lines[2] == "" and lines[3].endswith("**** New Scan · b ****") and lines[-1] == ""


def test_the_panel_dates_significant_events_and_warnings_but_not_routine_lines():
    panel = EventFormatter(LOG_FORMAT, "%H:%M:%S")
    assert panel.format(_record("scrolled")) == "15:40:12 INFO xdart.test: scrolled"
    assert panel.format(_record("rebuilt", **{log_events.EVENT: True})) == f"{_DATE} INFO xdart.test: rebuilt"
    assert panel.format(_record("refused", level=logging.WARNING)) == f"{_DATE} WARNING xdart.test: refused"
    # every line of the log file is dated already
    assert EventFormatter(LOG_FORMAT).format(_record("scrolled")).startswith(f"{_DATE},000 INFO")


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


@pytest.mark.parametrize("flag", [None, "XDART_PERF", "XDART_PERF_QUARTILES"])
def test_diagnostics_reach_the_regular_log_only_while_performance_monitoring_is_on(monkeypatch, caplog, flag):
    for name in ("XDART_PERF", "XDART_PERF_QUARTILES"):
        monkeypatch.delenv(name, raising=False)
    if flag is not None:
        monkeypatch.setenv(flag, "1")
    logger = logging.getLogger("xdart.test.diagnostic")
    caplog.set_level(logging.DEBUG, logger=logger.name)
    log_events.diagnostic(logger, "[PERF-SUMMARY] frames=%d", 3)
    log_events.diagnostic(logger, "Total Time: 1.00s", shown=False)  # a Run's own reading, taken at its start
    log_events.diagnostic(logger, "Total Frames Processed: 3", shown=True)
    assert [(record.levelno, record.getMessage()) for record in caplog.records] == [
        (logging.DEBUG if flag is None else logging.INFO, "[PERF-SUMMARY] frames=3"),
        (logging.DEBUG, "Total Time: 1.00s"), (logging.INFO, "Total Frames Processed: 3")]
    assert log_events.performance_monitoring() is (flag is not None)
