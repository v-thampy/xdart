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


class _FakeReadBlock:
    def __init__(self, array):
        self.array = array


class _FakeCursor:
    """Minimal stand-in for the R2 ContainerCursor read backend (the wrangler's
    h5py-backed read path now goes through the cursor, not a raw dataset)."""

    def __init__(self, data, is_2d=False):
        self._data = np.asarray(data)
        self.frame_count = int(self._data.shape[0])
        self.is_2d = is_2d

    def read_frame(self, idx):
        return np.asarray(self._data[int(idx)])

    def read_block(self, start, stop):
        return _FakeReadBlock(np.asarray(self._data[int(start):int(stop)]))

    def close(self):
        pass


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
    worker = _bare_image_thread()
    worker._eiger_master_path = str(tmp_path / "scan_master.h5")
    worker._eiger_frame_idx = 0
    worker._eiger_nframes = 1
    worker._eiger_master_queue = deque()
    worker._eiger_done_masters = set()
    worker._eiger_fabio_handle = None
    # R2: the h5py-backed single-frame read now goes through the cursor.
    worker._eiger_cursor = _FakeCursor(
        np.arange(4, dtype=np.uint16).reshape(1, 2, 2))
    worker._eiger_provider = None
    worker.inp_type = "Image File"

    _path, _scan_name, _number, image, _meta = worker._get_next_eiger_frame_sync()

    assert image.dtype == np.uint16


def test_prefetch_bulk_read_keeps_native_dataset_dtype(tmp_path):
    worker = _bare_image_thread()
    worker.command = ""
    worker._prefetch_stop_evt = threading.Event()
    worker._prefetch_queue = queue.Queue(maxsize=8)
    worker._prefetch_error = None
    worker._perf = None
    worker._eiger_master_path = str(tmp_path / "scan_master.h5")
    worker._eiger_frame_idx = 0
    worker._eiger_nframes = 3
    # R2: the bulk read now goes through the cursor's read_block, sized by the
    # ReadPlan's block_frames instead of the retired fixed-16 constant.
    worker._eiger_cursor = _FakeCursor(
        np.stack([np.full((2, 2), v, dtype=np.uint16) for v in range(3)]))
    worker._eiger_read_plan = SimpleNamespace(block_frames=16)
    worker._eiger_provider = None
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


def test_prefetch_retains_at_most_one_owner_block(tmp_path):
    """R2-R2: the prefetch worker copies each frame out of its native owner
    block, so the block is released as soon as its frames are dispatched — at
    most ONE source-owner block is ever live, never two (which would double the
    single-block byte budget).  Fail-before (queuing views) pinned every block
    the queue still held, so multiple owner blocks were live at once."""
    import gc
    import weakref

    N = 8
    data = np.arange(N * 4 * 4, dtype=np.uint16).reshape(N, 4, 4)
    block_refs = []

    class TrackingCursor:
        frame_count = N
        is_2d = False

        def read_block(self, start, stop):
            arr = np.array(data[int(start):int(stop)])   # a fresh native owner block
            block_refs.append(weakref.ref(arr))
            return _FakeReadBlock(arr)

        def read_frame(self, idx):
            return np.array(data[int(idx)])

        def close(self):
            pass

    worker = _bare_image_thread()
    worker.command = ""
    worker._prefetch_stop_evt = threading.Event()
    # A generous queue with NO consumer is the WORST case for owner retention:
    # every dispatched frame stays queued, so a view would pin every block.
    worker._prefetch_queue = queue.Queue(maxsize=1000)
    worker._prefetch_error = None
    worker._perf = None
    worker._eiger_master_path = str(tmp_path / "m.nxs")
    worker._eiger_frame_idx = 0
    worker._eiger_nframes = N
    worker._eiger_cursor = TrackingCursor()
    worker._eiger_read_plan = SimpleNamespace(block_frames=2)  # 2-frame blocks
    worker._eiger_provider = None
    worker._eiger_fabio_handle = None

    calls = {"n": 0}

    def fake_sync():
        calls["n"] += 1
        if calls["n"] == 1:
            worker._eiger_frame_idx = 1
            return (worker._eiger_master_path, "scan", 1,
                    np.zeros((4, 4), np.uint16), {})
        return (None, None, 1, None, {})

    worker._get_next_eiger_frame_sync = fake_sync

    max_live = [0]
    orig_push = worker._push_frame_to_queue

    def sampling_push(item, **kwargs):
        gc.collect()
        n_live = sum(1 for ref in block_refs if ref() is not None)
        max_live[0] = max(max_live[0], n_live)
        return orig_push(item, **kwargs)

    worker._push_frame_to_queue = sampling_push
    worker._prefetch_worker()

    assert max_live[0] <= 1, (
        f"retained {max_live[0]} simultaneous native owner blocks (budget is one)")
    # every frame was still dispatched (frame 1 via sync + 2..N via blocks)
    queued = []
    while not worker._prefetch_queue.empty():
        it = worker._prefetch_queue.get_nowait()
        if it[3] is not None:
            queued.append(it[2])
    assert queued == list(range(1, N + 1))
    # after the worker drains, no owner block is retained at all
    gc.collect()
    assert all(ref() is None for ref in block_refs)


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
