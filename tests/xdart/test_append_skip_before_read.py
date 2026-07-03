from __future__ import annotations

import logging
import os
import queue
import threading
import importlib.util
from collections import Counter, deque
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_HAS_PYQTGRAPH = importlib.util.find_spec("pyqtgraph") is not None
pytestmark = pytest.mark.skipif(
    not _HAS_PYQTGRAPH,
    reason="pyqtgraph GUI dependency is not installed",
)

if _HAS_PYQTGRAPH:
    from xdart.gui.tabs.static_scan.wranglers import image_wrangler_thread as iwt
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import imageThread


def test_paths_with_suffix_matches_extensions_case_insensitively(tmp_path):
    (tmp_path / "scan_0001.TIF").touch()
    (tmp_path / "scan_0002.tif").touch()
    (tmp_path / "scan_master.HDF5").touch()
    (tmp_path / "notes.txt").touch()

    tif_names = sorted(p.name for p in iwt._paths_with_suffix(tmp_path, ".tif"))
    hdf5_names = sorted(p.name for p in iwt._paths_with_suffix(tmp_path, "_master.hdf5"))

    assert tif_names == ["scan_0001.TIF", "scan_0002.tif"]
    assert hdf5_names == ["scan_master.HDF5"]


def _bare_worker(tmp_path):
    worker = imageThread.__new__(imageThread)
    worker.write_mode = "Append"
    worker.xye_only = False
    worker.series_average = False
    worker.h5_dir = str(tmp_path / "out")
    worker.file_lock = threading.RLock()
    worker.scan_args = {}
    worker.gi = False
    worker.incidence_motor = None
    worker.single_img = False
    worker.img_file = ""
    worker.img_ext = "tif"
    worker.img_dir = str(tmp_path)
    worker.inp_type = "Image File"
    worker.include_subdir = False
    worker.file_filter = ""
    worker.mask = None
    worker.detector_shape = None
    worker.meta_ext = None
    worker.meta_dir = None
    worker.command = ""
    worker.showLabel = SimpleNamespace(emit=lambda *_: None)
    worker._append_skip_frames_by_scan = {}
    worker._append_skip_without_reading = 0
    worker._discovered_frame_count = 0
    worker._skip_reason_counts = Counter()
    worker._append_skip_snapshot_warnings = set()
    return worker


class _FakeSnapshotScan:
    frame_index = [1, 2]
    load_calls = []

    def __init__(self, name, data_file=None, file_lock=None, **_kwargs):
        self.name = name
        self.data_file = data_file
        self.file_lock = file_lock
        self.scan_lock = threading.RLock()
        self.frames = SimpleNamespace(index=[])

    def load_from_h5(self, replace=False, mode="r"):
        self.load_calls.append((self.name, self.data_file, replace, mode))
        self.frames.index = list(self.frame_index)


def test_append_image_series_skips_before_reader_and_metadata(monkeypatch, tmp_path):
    paths = [tmp_path / f"scan_{idx:04d}.tif" for idx in (1, 2, 3)]
    for path in paths:
        path.touch()

    worker = _bare_worker(tmp_path)
    worker.inp_type = "Image Series"
    worker.img_file = str(paths[0])
    worker.img_dir = str(tmp_path)
    worker.img_ext = "tif"
    worker.scan_name = "scan"
    worker.img_fnames = []
    worker.processed = []
    worker.meta_ext = "txt"
    worker._append_skip_frames_by_scan = {"scan": {1, 2}}

    read_calls = []
    meta_calls = []

    def fake_read(path):
        read_calls.append(os.fspath(path))
        return np.ones((2, 2), dtype=float)

    def fake_meta(path, *, meta_format=None, meta_dir=None):
        meta_calls.append(os.fspath(path))
        return {"ok": 1}

    monkeypatch.setattr(iwt, "read_image", fake_read)
    monkeypatch.setattr(iwt, "read_image_metadata", fake_meta)

    img_file, scan_name, img_number, img_data, img_meta = worker.get_next_image()

    assert img_file == str(paths[2])
    assert scan_name == "scan"
    assert img_number == 3
    assert img_data.shape == (2, 2)
    assert img_meta == {"ok": 1}
    assert read_calls == [str(paths[2])]
    assert meta_calls == [str(paths[2])]
    assert worker._append_skip_without_reading == 2


def test_single_image_append_skip_emits_update_without_reader(monkeypatch, tmp_path):
    image = tmp_path / "scan_0001.tif"
    image.touch()

    worker = _bare_worker(tmp_path)
    worker.single_img = True
    worker.img_file = str(image)
    worker._append_skip_frames_by_scan = {"scan": {1}}
    updates = []
    worker.sigUpdate = SimpleNamespace(emit=updates.append)

    monkeypatch.setattr(
        iwt,
        "read_image",
        lambda _path: pytest.fail("already-indexed single image was read"),
    )

    _img_file, scan_name, img_number, img_data, _meta = worker.get_next_image()

    assert scan_name == "scan"
    assert img_number == 1
    assert img_data is None
    assert updates == [1]


def test_dash_index_filenames_enumerate_and_zero_is_valid(tmp_path):
    base = "P25_C5_eta0p025_scan1FRFR_d1200"
    files = [
        tmp_path / f"{base}-00000.tif",
        tmp_path / f"{base}-00001.tif",
        tmp_path / f"{base}_00002.tif",
    ]
    for path in files:
        path.touch()
    (tmp_path / f"{base}_extra_00003.tif").touch()

    assert iwt._get_scan_info(files[0]) == (base, 0)
    assert iwt._get_scan_info(tmp_path / "scan_0001.tif") == ("scan", 1)

    worker = _bare_worker(tmp_path)
    worker.inp_type = "Image Series"
    worker.img_file = str(files[0])
    worker.img_dir = str(tmp_path)
    worker.img_ext = "tif"
    worker.scan_name = base

    enumerated = worker._enumerate_scan_files()

    assert [(os.path.basename(path), num) for path, num in enumerated] == [
        (files[0].name, 0),
        (files[1].name, 1),
        (files[2].name, 2),
    ]


def test_eiger_prefetch_skips_indexed_frames_without_dataset_read(tmp_path):
    class FakeId:
        def refresh(self):
            return None

    class FakeDataset:
        shape = (3, 2, 2)
        id = FakeId()

        def __init__(self):
            self.calls = []

        def __getitem__(self, key):
            self.calls.append(key)
            return np.full((2, 2), int(key), dtype=np.uint16)

    dataset = FakeDataset()
    worker = _bare_worker(tmp_path)
    worker.command = ""
    worker.inp_type = "Image File"
    worker.img_file = str(tmp_path / "scan_master.h5")
    worker._eiger_master_path = worker.img_file
    worker._eiger_frame_idx = 0
    worker._eiger_nframes = 3
    worker._eiger_h5_dataset = dataset
    worker._eiger_h5_handle = None
    worker._eiger_fabio_handle = None
    worker._eiger_master_queue = deque()
    worker._eiger_done_masters = set()
    worker._eiger_metadata_cache = {}
    worker._prefetch_stop_evt = threading.Event()
    worker._prefetch_queue = queue.Queue(maxsize=8)
    worker._prefetch_error = None
    worker._perf = None
    worker._append_skip_frames_by_scan = {"scan": {1, 2, 3}}
    worker._read_eiger_metadata = lambda _path: pytest.fail(
        "metadata read should be skipped with indexed frames"
    )

    worker._prefetch_worker()

    queued = []
    while not worker._prefetch_queue.empty():
        queued.append(worker._prefetch_queue.get_nowait())

    assert dataset.calls == []
    assert queued == [(None, None, 1, None, {})]
    assert worker._append_skip_without_reading == 3


def test_append_skip_snapshot_lookup_never_opens_disk(monkeypatch, tmp_path):
    worker = _bare_worker(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "scan.nxs").touch()

    def fail_live_scan(*_args, **_kwargs):
        pytest.fail("per-frame append skip lookup opened the .nxs")

    monkeypatch.setattr(iwt, "LiveScan", fail_live_scan)

    assert worker._should_skip_before_read("scan", 1) is False
    assert worker._append_skip_frames_by_scan == {"scan": set()}


def test_append_skip_snapshot_primes_once_read_only(monkeypatch, tmp_path):
    worker = _bare_worker(tmp_path)
    worker.inp_type = "Image File"
    worker.img_file = str(tmp_path / "scan_master.h5")
    worker.img_ext = "h5"
    worker.scan_name = "scan_master"
    out = tmp_path / "out"
    out.mkdir()
    output = out / "scan.nxs"
    output.touch()

    _FakeSnapshotScan.load_calls = []
    _FakeSnapshotScan.frame_index = [1, 2]
    monkeypatch.setattr(iwt, "LiveScan", _FakeSnapshotScan)

    worker._prime_append_skip_snapshots_for_run()
    worker._prime_append_skip_snapshots_for_run()

    assert _FakeSnapshotScan.load_calls == [
        ("scan", str(output), False, "r")
    ]
    assert worker._append_skip_frames_by_scan == {"scan": {1, 2}}
    assert worker.fname == str(output)
    assert worker._should_skip_before_read("scan", 2) is True


def test_append_snapshot_primes_with_read_handle_open(monkeypatch, tmp_path):
    import h5py

    worker = _bare_worker(tmp_path)
    worker.inp_type = "Image File"
    worker.img_file = str(tmp_path / "scan_master.h5")
    worker.img_ext = "h5"
    out = tmp_path / "out"
    out.mkdir()
    output = out / "scan.nxs"
    with h5py.File(output, "w") as h5:
        h5.create_group("entry")

    class H5OpeningSnapshotScan(_FakeSnapshotScan):
        frame_index = [4, 5]

        def load_from_h5(self, replace=False, mode="r"):
            self.load_calls.append((self.name, self.data_file, replace, mode))
            with h5py.File(self.data_file, mode):
                pass
            self.frames.index = list(self.frame_index)

    H5OpeningSnapshotScan.load_calls = []
    monkeypatch.setattr(iwt, "LiveScan", H5OpeningSnapshotScan)

    with h5py.File(output, "r"):
        worker._prime_append_skip_snapshots_for_run()

    assert H5OpeningSnapshotScan.load_calls == [
        ("scan", str(output), False, "r")
    ]
    assert worker._append_skip_frames_by_scan == {"scan": {4, 5}}


def test_append_fresh_scan_primes_empty_and_reads_all(monkeypatch, tmp_path):
    paths = [tmp_path / f"scan_{idx:04d}.tif" for idx in (1, 2)]
    for path in paths:
        path.touch()

    worker = _bare_worker(tmp_path)
    worker.inp_type = "Image Series"
    worker.img_file = str(paths[0])
    worker.img_dir = str(tmp_path)
    worker.img_ext = "tif"
    worker.scan_name = "scan"
    worker.img_fnames = []
    worker.processed = []

    def fail_live_scan(*_args, **_kwargs):
        pytest.fail("fresh append snapshot should not load a missing .nxs")

    read_calls = []

    def fake_read(path):
        read_calls.append(os.fspath(path))
        return np.ones((2, 2), dtype=float)

    monkeypatch.setattr(iwt, "LiveScan", fail_live_scan)
    monkeypatch.setattr(iwt, "read_image", fake_read)

    worker._prime_append_skip_snapshots_for_run()
    img_file, scan_name, img_number, img_data, _meta = worker.get_next_image()

    assert worker._append_skip_frames_by_scan == {"scan": set()}
    assert worker.fname == str(tmp_path / "out" / "scan.nxs")
    assert img_file == str(paths[0])
    assert scan_name == "scan"
    assert img_number == 1
    assert img_data.shape == (2, 2)
    assert read_calls == [str(paths[0])]


def test_append_snapshot_failure_warns_once_and_skips_nothing(
        monkeypatch, caplog, tmp_path):
    worker = _bare_worker(tmp_path)
    worker.inp_type = "Image File"
    worker.img_file = str(tmp_path / "scan_master.h5")
    worker.img_ext = "h5"
    out = tmp_path / "out"
    out.mkdir()
    (out / "scan.nxs").touch()

    class BrokenSnapshotScan(_FakeSnapshotScan):
        def load_from_h5(self, replace=False, mode="r"):
            raise OSError("held read handle")

    monkeypatch.setattr(iwt, "LiveScan", BrokenSnapshotScan)

    with caplog.at_level(logging.WARNING):
        worker._prime_append_skip_snapshots_for_run()
        worker._prime_append_skip_snapshots_for_run()

    assert worker._append_skip_frames_by_scan == {"scan": set()}
    assert worker._should_skip_before_read("scan", 1) is False
    warnings = [
        rec for rec in caplog.records
        if "append skip snapshot unavailable" in rec.message
    ]
    assert len(warnings) == 1


def test_zero_processed_already_processed_frames_logs_info(caplog, tmp_path):
    worker = _bare_worker(tmp_path)
    labels = []
    worker.showLabel = SimpleNamespace(emit=labels.append)
    worker._record_discovered_frame()
    worker._record_skip_reason("already processed")

    with caplog.at_level(logging.INFO):
        worker._report_run_skip_summary(0)

    expected = "0 of 1 discovered frame(s) processed: already processed"
    assert expected in caplog.text
    assert not [
        rec for rec in caplog.records
        if rec.levelno >= logging.WARNING and expected in rec.message
    ]
    assert labels == [expected]


# ── MEM-1c: series-average + Append must not silently produce nothing ────────
def test_series_average_append_refuses_when_averaged_output_exists(
        monkeypatch, tmp_path):
    worker = _bare_worker(tmp_path)
    worker.series_average = True           # collapses every source frame -> #1
    monkeypatch.setattr(worker, "_append_run_start_scan_names", lambda: ["scan"])
    worker._append_skip_frames_by_scan = {"scan": {1}}   # averaged output on disk

    blocker = worker._series_average_append_blocker()
    assert blocker is not None
    assert "already exists" in blocker
    assert "Replace" in blocker            # actionable remedy


def test_series_average_append_allowed_when_output_absent(monkeypatch, tmp_path):
    worker = _bare_worker(tmp_path)
    worker.series_average = True
    monkeypatch.setattr(worker, "_append_run_start_scan_names", lambda: ["scan"])
    worker._append_skip_frames_by_scan = {"scan": set()}  # fresh target

    assert worker._series_average_append_blocker() is None


def test_non_series_average_append_never_blocked(monkeypatch, tmp_path):
    worker = _bare_worker(tmp_path)        # series_average=False (default)
    monkeypatch.setattr(worker, "_append_run_start_scan_names", lambda: ["scan"])
    worker._append_skip_frames_by_scan = {"scan": {1, 2, 3}}

    assert worker._series_average_append_blocker() is None


def test_series_average_replace_mode_not_blocked(monkeypatch, tmp_path):
    worker = _bare_worker(tmp_path)
    worker.series_average = True
    worker.write_mode = "Overwrite"        # not Append => append-skip disabled
    monkeypatch.setattr(worker, "_append_run_start_scan_names", lambda: ["scan"])
    worker._append_skip_frames_by_scan = {"scan": {1}}

    assert worker._series_average_append_blocker() is None


def test_zero_processed_real_failure_still_warns(caplog, tmp_path):
    worker = _bare_worker(tmp_path)
    labels = []
    worker.showLabel = SimpleNamespace(emit=labels.append)
    worker._record_discovered_frame()
    worker._record_skip_reason("unreadable or empty image data")

    with caplog.at_level(logging.WARNING):
        worker._report_run_skip_summary(0)

    expected = (
        "0 of 1 discovered frame(s) processed: unreadable or empty image data"
    )
    assert expected in caplog.text
    assert [
        rec for rec in caplog.records
        if rec.levelno >= logging.WARNING and expected in rec.message
    ]
    assert labels == [expected]
