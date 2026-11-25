"""Drive the mounted Run bar and the real Live cadence with a controlled clock."""

from importlib.util import find_spec
from time import monotonic

import pytest
from pyqtgraph.Qt import QtTest


class MissingLiveCapability(AssertionError):
    """The mounted page on the implementation base has no per-scan Live affordance."""


unimplemented_live = pytest.mark.xfail(
    find_spec("xdart.gui.widgets.live_scan_watcher") is None,
    reason="the implementation base hides per-scan Live on the native page",
    raises=MissingLiveCapability,
    strict=True,
)


class LiveClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds=2.0):
        self.now += seconds


def wait_until(qapp, predicate, *, timeout_ms=15000):
    deadline = monotonic() + timeout_ms / 1000
    while not predicate() and monotonic() < deadline:
        qapp.processEvents()
        QtTest.QTest.qWait(5)
    assert predicate(), "the expected native owner or Qt state never arrived"


def start_live(page):
    bar = page.run_bar
    if bar.liveButton.isHidden():
        raise MissingLiveCapability(f"{type(page).__name__} hides the mounted Live control")
    watcher = page._live_watcher
    watcher.clock = LiveClock()
    assert bar.liveButton.isEnabled()
    if not bar.is_live():
        bar.liveButton.click()
    assert bar.is_live()
    assert bar.startButton.isEnabled(), bar.readinessLabel.full_text()
    bar.startButton.click()
    return watcher


def tick_live(page, qapp, seconds=2.0):
    watcher = page._live_watcher
    if watcher.polling:
        wait_until(qapp, lambda: not watcher.polling)
    watcher.clock.advance(seconds)
    watcher.poll_due()
    wait_until(qapp, lambda: not watcher.polling)
