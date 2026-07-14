from __future__ import annotations

import os
import queue
import threading
from collections import deque
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import imageThread


def _bare_image_thread():
    worker = imageThread.__new__(imageThread)
    worker.meta_ext = None
    worker.meta_dir = None
    worker._eiger_metadata_cache = {}
    return worker


def test_eiger_metadata_is_cached_per_master(monkeypatch, tmp_path):
    from xdart.gui.tabs.static_scan.wranglers import image_wrangler_thread

    calls = []

    def fake_read(path, *, meta_format=None, meta_dir=None):
        calls.append((os.fspath(path), meta_format, meta_dir))
        return {"theta": 1.25}

    monkeypatch.setattr(image_wrangler_thread, "read_image_metadata", fake_read)

    worker = _bare_image_thread()
    worker.meta_ext = "txt"
    worker.meta_dir = str(tmp_path / "meta")
    master = tmp_path / "scan_master.h5"

    first = worker._read_eiger_metadata(master)
    first["theta"] = 99.0
    second = worker._read_eiger_metadata(master)

    assert calls == [(str(master), "txt", str(tmp_path / "meta"))]
    assert second == {"theta": 1.25}


def test_sync_eiger_read_keeps_native_dataset_dtype(tmp_path):
    class FakeDataset:
        shape = (1, 2, 2)
        ndim = 3     # a real h5py.Dataset always carries ndim (F6 2-D branch)

        def __getitem__(self, key):
            assert key == 0
            return np.arange(4, dtype=np.uint16).reshape(2, 2)

    worker = _bare_image_thread()
    worker._eiger_master_path = str(tmp_path / "scan_master.h5")
    worker._eiger_frame_idx = 0
    worker._eiger_nframes = 1
    worker._eiger_master_queue = deque()
    worker._eiger_done_masters = set()
    worker._eiger_h5_handle = None
    worker._eiger_h5_dataset = FakeDataset()
    worker._eiger_fabio_handle = None
    worker.inp_type = "Image File"

    _path, _scan_name, _number, image, _meta = worker._get_next_eiger_frame_sync()

    assert image.dtype == np.uint16


def test_prefetch_bulk_read_keeps_native_dataset_dtype(tmp_path):
    class FakeDataset:
        shape = (3, 2, 2)
        ndim = 3     # a real h5py.Dataset always carries ndim (F6 2-D branch)

        def __getitem__(self, key):
            if isinstance(key, slice):
                frames = [
                    np.full((2, 2), value, dtype=np.uint16)
                    for value in range(key.start, key.stop)
                ]
                return np.stack(frames, axis=0)
            return np.full((2, 2), key, dtype=np.uint16)

    worker = _bare_image_thread()
    worker.command = ""
    worker._prefetch_stop_evt = threading.Event()
    worker._prefetch_queue = queue.Queue(maxsize=8)
    worker._prefetch_error = None
    worker._perf = None
    worker._eiger_master_path = str(tmp_path / "scan_master.h5")
    worker._eiger_frame_idx = 0
    worker._eiger_nframes = 3
    worker._eiger_h5_dataset = FakeDataset()
    worker._eiger_fabio_handle = None

    calls = 0

    def fake_sync_read():
        nonlocal calls
        calls += 1
        if calls == 1:
            worker._eiger_frame_idx = 1
            return (
                worker._eiger_master_path,
                "scan",
                1,
                np.zeros((2, 2), dtype=np.uint16),
                {},
            )
        return (None, None, 1, None, {})

    worker._get_next_eiger_frame_sync = fake_sync_read

    worker._prefetch_worker()

    queued = []
    while not worker._prefetch_queue.empty():
        queued.append(worker._prefetch_queue.get_nowait())

    assert [item[2] for item in queued] == [1, 2, 3, 1]
    assert queued[1][3].dtype == np.uint16
    assert queued[2][3].dtype == np.uint16


def test_prefetch_worker_keeps_generation_handles_during_cleanup(tmp_path):
    """A timed-out Stop cleanup may clear public handles while the old reader
    is still returning from HDF5.  The worker must finish through its own event
    and queue, never dereference the replaced ``None`` attributes."""
    worker = _bare_image_thread()
    worker.command = ""
    prefetch_queue = queue.Queue(maxsize=4)
    stop_evt = threading.Event()
    worker._prefetch_queue = prefetch_queue
    worker._prefetch_stop_evt = stop_evt
    worker._prefetch_error = None
    worker._perf = None

    def slow_read_returning_after_cleanup():
        worker._prefetch_queue = None
        worker._prefetch_stop_evt = None
        return (None, None, 1, None, {})

    worker._get_next_eiger_frame_sync = slow_read_returning_after_cleanup

    worker._prefetch_worker(prefetch_queue, stop_evt)

    assert prefetch_queue.get_nowait() == (None, None, 1, None, {})


def test_prefetch_cleanup_reports_reader_that_outlives_join():
    class StillAlive:
        def __init__(self):
            self.join_calls = []

        def is_alive(self):
            return True

        def join(self, timeout=None):
            self.join_calls.append(timeout)

    worker = _bare_image_thread()
    stop_evt = threading.Event()
    prefetch_queue = queue.Queue()
    prefetch_queue.put(("queued",))
    thread = StillAlive()
    worker._prefetch_stop_evt = stop_evt
    worker._prefetch_queue = prefetch_queue
    worker._prefetch_thread = thread

    assert worker._prefetch_stop_prior() is False
    assert stop_evt.is_set()
    assert prefetch_queue.empty()
    assert thread.join_calls == [2.0]


def test_prefetch_cleanup_reports_reader_stopped_by_join():
    class StopsOnJoin:
        alive = True

        def is_alive(self):
            return self.alive

        def join(self, timeout=None):
            assert timeout == 2.0
            self.alive = False

    worker = _bare_image_thread()
    worker._prefetch_stop_evt = threading.Event()
    worker._prefetch_queue = queue.Queue()
    worker._prefetch_thread = StopsOnJoin()

    assert worker._prefetch_stop_prior() is True


def test_run_refusal_preserves_lingering_reader_state():
    class StillAlive:
        def is_alive(self):
            return True

        def join(self, timeout=None):
            assert timeout == 2.0

    worker = _bare_image_thread()
    worker.poni = object()
    worker.img_file = "raw.nxs"
    worker.command = "start"
    worker.showLabel = SimpleNamespace(emit=lambda message: statuses.append(message))
    worker.img_fnames = ["pending-image"]
    worker.processed = ["processed-image"]
    worker._frame_read_clocks = {"raw.nxs": 1.0}
    worker.processed_scans = ["scan"]
    worker._eiger_master_path = "active-master.nxs"
    worker._eiger_frame_idx = 7
    worker._eiger_nframes = 11
    worker._eiger_master_queue = deque(["next-master.nxs"])
    worker._eiger_done_masters = {"done-master.nxs"}
    worker._eiger_retry_after = {"retry-master.nxs": 2.0}
    worker._eiger_zero_frame_seen = {"shell.nxs": (1.0, 2.0)}
    worker._eiger_metadata_cache = {"active-master.nxs": {"hy": 1.0}}
    worker._bluesky_source_cache = {"active-master.nxs": object()}
    prefetch_queue = queue.Queue()
    prefetch_queue.put(("queued-frame",))
    worker._prefetch_queue = prefetch_queue
    worker._prefetch_stop_evt = threading.Event()
    worker._prefetch_thread = StillAlive()
    worker._eiger_close_master = lambda: (_ for _ in ()).throw(
        AssertionError("refused restart closed the old reader's handle"))
    statuses = []

    worker.run()

    assert worker.command == "stop"
    assert statuses == ["Previous detector reader is still stopping; retry Run shortly"]
    assert worker.img_fnames == ["pending-image"]
    assert worker.processed == ["processed-image"]
    assert worker._frame_read_clocks == {"raw.nxs": 1.0}
    assert worker.processed_scans == ["scan"]
    assert worker._eiger_master_path == "active-master.nxs"
    assert (worker._eiger_frame_idx, worker._eiger_nframes) == (7, 11)
    assert list(worker._eiger_master_queue) == ["next-master.nxs"]
    assert worker._eiger_done_masters == {"done-master.nxs"}
    assert worker._eiger_retry_after == {"retry-master.nxs": 2.0}
    assert worker._eiger_zero_frame_seen == {"shell.nxs": (1.0, 2.0)}
    assert worker._eiger_metadata_cache == {
        "active-master.nxs": {"hy": 1.0}}
    assert set(worker._bluesky_source_cache) == {"active-master.nxs"}
    assert worker._prefetch_queue is prefetch_queue
