# -*- coding: utf-8 -*-
"""Greenfield Phase 3 / D2: the background frame-hydration worker pulls evicted
publications off the GUI thread and signals completion."""
import threading
import time

import pytest

from pyqtgraph import Qt
from xdart.gui.tabs.static_scan.frame_hydration_worker import FrameHydrationWorker

_DIRECT = Qt.QtCore.Qt.ConnectionType.DirectConnection


@pytest.fixture(scope="module")
def qapp():
    from pyqtgraph.Qt import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def test_worker_hydrates_off_the_calling_thread_and_echoes_generation(qapp):
    caller_thread = threading.get_ident()
    seen = {}

    class FakeStore:
        def get_or_hydrate(self, label):
            seen["thread"] = threading.get_ident()
            return {"label": label}        # any non-None payload

    done = threading.Event()
    emitted = []
    worker = FrameHydrationWorker(FakeStore())
    # DirectConnection -> the slot runs ON the worker thread, so the test needs
    # no Qt event loop to observe the emission.
    worker.sigHydrated.connect(
        lambda label, gen: (emitted.append((label, gen)), done.set()), _DIRECT)
    worker.start()
    try:
        worker.request(7, 3)
        assert done.wait(5.0), "worker never emitted sigHydrated"
        assert emitted == [(7, 3)]                 # label + generation echoed
        assert seen["thread"] != caller_thread     # the read ran OFF the caller
    finally:
        worker.stop()
    assert not worker.isRunning()


def test_worker_does_not_emit_when_hydrate_yields_none(qapp):
    class NoneStore:
        def get_or_hydrate(self, label):
            return None

    emitted = []
    worker = FrameHydrationWorker(NoneStore())
    worker.sigHydrated.connect(lambda l, g: emitted.append((l, g)), _DIRECT)
    worker.start()
    try:
        worker.request(1, 1)
        time.sleep(0.3)
    finally:
        worker.stop()
    assert emitted == []


def test_worker_survives_a_raising_hydrator(qapp):
    class BoomStore:
        def get_or_hydrate(self, label):
            raise RuntimeError("disk gone")

    emitted = []
    worker = FrameHydrationWorker(BoomStore())
    worker.sigHydrated.connect(lambda l, g: emitted.append((l, g)), _DIRECT)
    worker.start()
    try:
        worker.request(1, 1)
        time.sleep(0.3)
        assert worker.isRunning()      # an exception in one request can't kill it
    finally:
        worker.stop()
    assert emitted == []


def test_request_after_stop_is_a_noop(qapp):
    class FakeStore:
        def get_or_hydrate(self, label):
            return {"label": label}

    emitted = []
    worker = FrameHydrationWorker(FakeStore())
    worker.sigHydrated.connect(lambda l, g: emitted.append((l, g)), _DIRECT)
    worker.start()
    worker.stop()
    worker.request(5, 1)               # after stop -> dropped
    time.sleep(0.2)
    assert emitted == []
