# -*- coding: utf-8 -*-
"""Coalescer — the one GUI coalescing idiom (Phase 1c)."""
from __future__ import annotations

import pytest
from pyqtgraph import Qt

from xdart.utils.throttle import Coalescer

QtCore = Qt.QtCore
QtWidgets = Qt.QtWidgets


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _spin_until(predicate, timeout_ms=2000):
    deadline = QtCore.QDeadlineTimer(timeout_ms)
    while not predicate() and not deadline.hasExpired():
        QtWidgets.QApplication.processEvents()
    return predicate()


def test_throttle_coalesces_but_keeps_streaming(qapp):
    """Throttle: a steady stream still fires every interval — the FIRST
    trigger's countdown is kept, not restarted (the freeze-until-end-of-
    scan debounce bug class)."""
    fired = []
    c = Coalescer(30, mode="throttle")
    c.triggered.connect(lambda: fired.append(len(fired)))

    c.trigger()
    for _ in range(20):          # burst within the interval
        c.trigger()
    assert c.is_pending()
    assert _spin_until(lambda: len(fired) >= 1)
    assert len(fired) == 1       # the burst coalesced into ONE emission

    c.trigger()                  # stream continues -> next interval fires
    assert _spin_until(lambda: len(fired) >= 2)


def test_debounce_restarts_until_quiet_then_fires_once(qapp):
    fired = []
    c = Coalescer(50, mode="debounce")
    c.triggered.connect(lambda: fired.append(1))
    for _ in range(5):
        c.trigger()              # each restarts the countdown
    assert c.is_pending()
    assert _spin_until(lambda: len(fired) == 1)
    QtWidgets.QApplication.processEvents()
    assert len(fired) == 1


def test_flush_fires_pending_now_and_cancel_drops_it(qapp):
    fired = []
    c = Coalescer(10_000, mode="debounce")   # would never fire in-test
    c.triggered.connect(lambda: fired.append(1))

    c.trigger()
    c.flush()
    assert fired == [1] and not c.is_pending()

    c.flush()                    # nothing pending: no double fire
    assert fired == [1]

    c.trigger()
    c.cancel()
    assert fired == [1] and not c.is_pending()


def test_qtimer_compatible_surface(qapp):
    """start/stop/isActive/setInterval keep bare-timer call sites (and
    test fakes written against them) working unchanged."""
    c = Coalescer(10_000, mode="throttle")
    assert not c.isActive()
    c.start()
    assert c.isActive()
    c.stop()
    assert not c.isActive()
    c.setInterval(25)
    assert c.interval() == 25


def test_mode_is_validated():
    with pytest.raises(ValueError, match="throttle.*debounce"):
        Coalescer(100, mode="bounce")
