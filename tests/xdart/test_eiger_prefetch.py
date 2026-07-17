from __future__ import annotations

import os
import queue
import threading
from collections import deque
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import (
    _PREFETCH_QUEUE_SIZE,
    imageThread,
)


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


class _BoundaryTrackingCursor:
    """R2-R2 (final): tracking cursor that samples live owner blocks INSIDE
    ``read_block`` — immediately after allocating the NEW block, i.e. at the
    exact allocation boundary where ``block = cursor.read_block(...).array``
    evaluates its right-hand side while the caller's prior ``block`` local
    still owns the previous array.  Sampling later (at queue-push, as the
    first version of this test did) misses that window because the assignment
    has already released the old owner by then."""

    is_2d = False

    def __init__(self, data, worker):
        import weakref

        self._data = np.asarray(data)
        self._worker = worker
        self.frame_count = int(self._data.shape[0])
        self.owner_refs = []       # (weakref, nbytes) of every owner block
        self.samples = []          # (live_owner_blocks, owner_bytes, queued_copy_bytes)
        self._weakref = weakref

    def _queued_copy_bytes(self):
        q = self._worker._prefetch_queue
        if q is None:
            return 0
        try:
            items = list(q.queue)
        except Exception:
            return 0
        return sum(it[3].nbytes for it in items
                   if it and len(it) > 3 and it[3] is not None)

    def read_block(self, start, stop):
        import gc

        arr = np.array(self._data[int(start):int(stop)])  # fresh native owner
        self.owner_refs.append((self._weakref.ref(arr), arr.nbytes))
        gc.collect()
        live = [(r, sz) for r, sz in self.owner_refs if r() is not None]
        self.samples.append((
            len(live),
            sum(sz for _r, sz in live),
            self._queued_copy_bytes(),
        ))
        return _FakeReadBlock(arr)

    def read_frame(self, idx):
        return np.array(self._data[int(idx)])

    def close(self):
        pass


def _owner_tracked_worker(tmp_path, n_frames, *, maxsize, block_frames=2):
    """A bare worker wired for the real ``_prefetch_worker`` with a
    boundary-sampling cursor, plus copy-weakref registration at the push
    seam (the first point a queued per-frame copy is observable)."""
    import weakref

    data = np.arange(n_frames * 4 * 4, dtype=np.uint16).reshape(n_frames, 4, 4)
    worker = _bare_image_thread()
    worker.command = ""
    worker._prefetch_stop_evt = threading.Event()
    worker._prefetch_queue = queue.Queue(maxsize=maxsize)
    worker._prefetch_error = None
    worker._perf = None
    worker._eiger_master_path = str(tmp_path / "m.nxs")
    worker._eiger_frame_idx = 0
    worker._eiger_nframes = n_frames
    cursor = _BoundaryTrackingCursor(data, worker)
    worker._eiger_cursor = cursor
    worker._eiger_read_plan = SimpleNamespace(block_frames=block_frames)
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

    copy_refs = []             # (weakref, nbytes) of every queued frame copy
    orig_push = worker._push_frame_to_queue

    def registering_push(item, **kwargs):
        if item and len(item) > 3 and item[3] is not None:
            copy_refs.append((weakref.ref(item[3]), item[3].nbytes))
        return orig_push(item, **kwargs)

    worker._push_frame_to_queue = registering_push
    return worker, cursor, copy_refs


def test_prefetch_releases_owner_block_before_next_allocation(tmp_path):
    """R2-R2 (final): the prior source-owner block must be EXPLICITLY released
    before the next ``read_block()`` allocation.  Fail-before (`8707982f`): the
    caller's ``block`` local still owned the previous array while the new one
    was allocated, so the in-``read_block`` sample saw TWO live owner blocks at
    every boundary after the first.  Pass-after: exactly one."""
    import gc

    N = 9  # sync frame 1 + four 2-frame blocks: [1:3] [3:5] [5:7] [7:9]
    worker, cursor, _copies = _owner_tracked_worker(
        tmp_path, N, maxsize=1000, block_frames=2)

    worker._prefetch_worker()

    assert len(cursor.samples) == 4, "expected 4 owner-block reads"
    peak = max(n for n, _b, _q in cursor.samples)
    assert peak <= 1, (
        f"{peak} simultaneous native owner blocks live at a read_block "
        f"allocation boundary (budget is one): samples={cursor.samples}")
    # every frame was still dispatched, in order
    queued = []
    while not worker._prefetch_queue.empty():
        it = worker._prefetch_queue.get_nowait()
        if it[3] is not None:
            queued.append(it[2])
    assert queued == list(range(1, N + 1))
    # after the worker drains, no owner block is retained at all
    gc.collect()
    assert all(r() is None for r, _sz in cursor.owner_refs)


def test_prefetch_owner_and_copy_bytes_bounded_with_production_queue(tmp_path):
    """R2-R2 (final): with the REAL production queue depth and a throttled
    consumer, the native-byte footprint stays bounded and separable:
    peak owner bytes <= one read-plan block; peak queued-copy bytes <= queue
    depth frames; combined live native bytes <= block + (depth + in-flight)
    frames.  Frame order is preserved and every owner block AND queued copy
    is released after the drain."""
    import gc
    import time as _time

    N = 9
    depth = _PREFETCH_QUEUE_SIZE  # the real production maxsize (default 4)
    assert depth == 4
    worker, cursor, copy_refs = _owner_tracked_worker(
        tmp_path, N, maxsize=depth, block_frames=2)
    frame_bytes = 4 * 4 * 2
    block_bytes = 2 * frame_bytes

    consumed = []

    def throttled_consumer():
        while True:
            it = worker._prefetch_queue.get()
            if it[3] is None:
                return
            consumed.append(it[2])
            it = None          # drop the copy ref before the next sleep
            _time.sleep(0.002)  # throttle: lets the queue back up to full

    consumer = threading.Thread(target=throttled_consumer, daemon=True)
    producer = threading.Thread(target=worker._prefetch_worker, daemon=True)
    producer.start()
    consumer.start()
    producer.join(timeout=30)
    consumer.join(timeout=30)
    assert not producer.is_alive() and not consumer.is_alive()

    assert consumed == list(range(1, N + 1))

    peak_owner_blocks = max(n for n, _b, _q in cursor.samples)
    peak_owner_bytes = max(b for _n, b, _q in cursor.samples)
    peak_queued_bytes = max(q for _n, _b, q in cursor.samples)
    peak_combined = max(b + q for _n, b, q in cursor.samples)
    assert peak_owner_blocks == 1
    assert peak_owner_bytes <= block_bytes
    assert peak_queued_bytes <= depth * frame_bytes
    # one block of owner + a full queue of copies (+1 frame for the copy the
    # worker holds in flight between building the item and the queue put)
    assert peak_combined <= block_bytes + (depth + 1) * frame_bytes

    # all owner blocks and ALL queued frame copies release after the drain
    gc.collect()
    assert all(r() is None for r, _sz in cursor.owner_refs)
    assert copy_refs and all(r() is None for r, _sz in copy_refs)


def test_prefetch_read_failure_fallback_releases_prior_owner(tmp_path):
    """R2-R2 (final): on a bulk read FAILURE the prior group's owner block has
    already been released (the explicit release runs at the end of each group,
    before the next allocation is even attempted), and the worker falls back
    to the sync reader without losing frames or order."""
    import gc

    N = 9
    worker, cursor, _copies = _owner_tracked_worker(
        tmp_path, N, maxsize=1000, block_frames=2)
    data = cursor._data

    # sequential sync reader: the fallback path re-serves frames one at a time
    def sequential_sync():
        idx = worker._eiger_frame_idx
        if idx >= N:
            return (None, None, 1, None, {})
        worker._eiger_frame_idx = idx + 1
        return (worker._eiger_master_path, "scan", idx + 1,
                np.array(data[idx]), {})

    worker._get_next_eiger_frame_sync = sequential_sync

    orig_read_block = cursor.read_block
    calls = {"n": 0}
    live_at_failure = []

    def failing_read_block(start, stop):
        calls["n"] += 1
        if calls["n"] == 2:  # fail the SECOND block read
            gc.collect()
            live_at_failure.append(sum(
                1 for r, _sz in cursor.owner_refs if r() is not None))
            raise OSError("injected block-read failure")
        return orig_read_block(start, stop)

    cursor.read_block = failing_read_block

    worker._prefetch_worker()

    # the failure-path entry saw NO lingering prior owner block
    assert live_at_failure == [0], (
        f"prior owner block still live entering the failing read_block: "
        f"{live_at_failure}")
    # the fallback still served every frame, in order, with no gaps
    queued = []
    while not worker._prefetch_queue.empty():
        it = worker._prefetch_queue.get_nowait()
        if it[3] is not None:
            queued.append(it[2])
    assert queued == list(range(1, N + 1))
    gc.collect()
    assert all(r() is None for r, _sz in cursor.owner_refs)


def test_prefetch_stop_releases_owner_and_queued_copies(tmp_path):
    """R2-R2 (final): Stop mid-stream (worker blocked pushing into a full
    queue, no consumer) must release every owner block and every queued copy
    once the worker exits and the generation queue is discarded."""
    import gc

    N = 9
    depth = _PREFETCH_QUEUE_SIZE
    worker, cursor, copy_refs = _owner_tracked_worker(
        tmp_path, N, maxsize=depth, block_frames=2)

    producer = threading.Thread(target=worker._prefetch_worker, daemon=True)
    producer.start()
    # wait until the worker is wedged on the full queue, then Stop
    deadline = 30.0
    while worker._prefetch_queue.qsize() < depth and deadline > 0:
        threading.Event().wait(0.005)
        deadline -= 0.005
    worker.command = "stop"
    worker._prefetch_stop_evt.set()
    producer.join(timeout=30)
    assert not producer.is_alive()

    # discard the generation queue exactly as cleanup does, then everything
    # (owner blocks AND the copies the queue still held) must be dead
    while not worker._prefetch_queue.empty():
        worker._prefetch_queue.get_nowait()
    worker._prefetch_queue = None
    gc.collect()
    assert cursor.owner_refs and all(r() is None for r, _sz in cursor.owner_refs)
    assert copy_refs and all(r() is None for r, _sz in copy_refs)


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
