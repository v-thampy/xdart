from __future__ import annotations

import os
import queue
import threading
from collections import deque

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
