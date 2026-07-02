# -*- coding: utf-8 -*-
"""Greenfield Phase 3 / D2: the background frame-hydration worker pulls evicted
publications off the GUI thread and signals completion."""
import threading
import time
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from pyqtgraph import Qt
from xdart.gui.tabs.static_scan.frame_hydration_worker import FrameHydrationWorker
from xrd_tools.core import (
    FrameRecord,
    FrameView,
    IntegrationResult1D,
    assert_frameview_equivalent,
    view_to_result_1d,
)
from xrd_tools.io import read_frame_view, write_integrated_stack
from xrd_tools.session import FrameRecordStore

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
        stopped = worker.stop()
    assert stopped is True                         # P1: stop reports success
    assert not worker.isRunning()


def test_worker_coalesces_superseded_generation(qapp):
    """P3: a request whose generation is older than the newest one queued is
    skipped WITHOUT hitting disk — the user already scrolled past it."""
    reads = []
    done = threading.Event()

    class FakeStore:
        def get_or_hydrate(self, label):
            reads.append(label)
            if label == 2:
                done.set()
            return {"label": label}

    worker = FrameHydrationWorker(FakeStore())
    # Enqueue BOTH before starting the loop: gen-1 is superseded by gen-2.
    worker.request(1, 1)
    worker.request(2, 2)
    worker.start()
    try:
        assert done.wait(5.0)
        time.sleep(0.1)                            # let a (wrong) gen-1 read happen
    finally:
        worker.stop()
    assert 2 in reads
    assert 1 not in reads, f"superseded gen-1 read should be skipped; got {reads}"


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


def test_worker_hydrates_each_store_from_provider(qapp):
    calls = []
    done = threading.Event()

    class Store:
        def __init__(self, name):
            self.name = name

        def get_or_hydrate(self, label):
            calls.append((self.name, label))
            return {"label": label}

    worker = FrameHydrationWorker(lambda: (Store("record"), Store("publication")))
    worker.sigHydrated.connect(lambda l, g: done.set(), _DIRECT)
    worker.start()
    try:
        worker.request(4, 1)
        assert done.wait(5.0)
    finally:
        worker.stop()
    assert calls == [("record", 4), ("publication", 4)]


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


def _view(label, scale=1.0):
    intensity = scale * np.array([3.0, 6.0, 12.0])
    return FrameView.from_results(
        label=label,
        result_1d=IntegrationResult1D(
            radial=np.array([0.1, 0.2, 0.3]),
            intensity=intensity,
            sigma=np.sqrt(intensity),
            unit="q_A^-1",
        ),
    )


def test_record_store_rehydrates_evicted_frame_on_worker_thread(qapp, tmp_path):
    path = tmp_path / "record_store_hydrate.nxs"
    original = _view(1, scale=1.0)
    second = _view(2, scale=2.0)
    with h5py.File(path, "w") as f:
        write_integrated_stack(
            f.create_group("entry"),
            frame_indices=[1, 2],
            results_1d=[view_to_result_1d(original), view_to_result_1d(second)],
        )

    store = FrameRecordStore(max_heavy_items=1)
    store.upsert(FrameRecord.from_view(original), persisted=True)
    store.upsert(FrameRecord.from_view(second), persisted=True)
    assert not store.has_heavy_payload(1)
    assert store.has_heavy_payload(2)

    caller_thread = threading.get_ident()
    seen = {}

    def hydrate(label):
        seen["thread"] = threading.get_ident()
        return FrameRecord.from_view(read_frame_view(path, int(label)))

    store.set_hydrator(hydrate)

    done = threading.Event()
    worker = FrameHydrationWorker(store)
    worker.sigHydrated.connect(lambda label, gen: done.set(), _DIRECT)
    worker.start()
    try:
        worker.request(1, 1)
        assert done.wait(5.0), "record-store hydration did not finish"
    finally:
        worker.stop()

    assert seen["thread"] != caller_thread
    assert store.has_heavy_payload(1)
    assert not store.has_heavy_payload(2)
    assert_frameview_equivalent(store.get(1).project(), original)


def test_record_store_hydrator_serializes_with_writer_file_lock(monkeypatch):
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import imageThread
    import xrd_tools.io as xrd_io

    file_lock = threading.Lock()
    store = FrameRecordStore(max_heavy_items=1)
    host = SimpleNamespace(
        file_lock=file_lock,
        _on_qt_gui_thread=lambda: False,
    )
    scan = SimpleNamespace(data_file="/tmp/live_scan.nxs")
    entered = threading.Event()

    def fake_read_frame_view(path, label, *, mode_1d=None, mode_2d=None):
        entered.set()
        return _view(label)

    monkeypatch.setattr(xrd_io, "read_frame_view", fake_read_frame_view)
    hydrate = imageThread._record_store_hydrator(host, scan, store)

    file_lock.acquire()
    result = []
    thread = threading.Thread(target=lambda: result.append(hydrate(1)))
    thread.start()
    try:
        assert not entered.wait(0.1), "hydration read overlapped writer lock"
    finally:
        file_lock.release()
    thread.join(2.0)

    assert not thread.is_alive()
    assert entered.is_set()
    assert isinstance(result[0], FrameRecord)


def test_record_store_hydrator_reads_immediately_when_writer_idle(monkeypatch):
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import imageThread
    import xrd_tools.io as xrd_io

    file_lock = threading.Lock()
    store = FrameRecordStore(max_heavy_items=1)
    host = SimpleNamespace(
        file_lock=file_lock,
        _on_qt_gui_thread=lambda: False,
    )
    scan = SimpleNamespace(data_file="/tmp/live_scan.nxs")
    entered = threading.Event()

    def fake_read_frame_view(path, label, *, mode_1d=None, mode_2d=None):
        entered.set()
        return _view(label)

    monkeypatch.setattr(xrd_io, "read_frame_view", fake_read_frame_view)
    hydrate = imageThread._record_store_hydrator(host, scan, store)

    result = hydrate(2)

    assert entered.is_set()
    assert isinstance(result, FrameRecord)
