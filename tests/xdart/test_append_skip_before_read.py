from __future__ import annotations

import logging
import os
import queue
import threading
import importlib.util
from collections import Counter, deque
from contextlib import contextmanager
from pathlib import Path
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


def _frozen_run_config(*, skip_2d=True, gi=False, bai_1d_args=None,
                       bai_2d_args=None, gi_mode_1d="q_total",
                       gi_mode_2d="qip_qoop", output_mode="Append",
                       live_mode=False):
    """A REAL ``FrozenRunConfiguration`` through the PRODUCTION freeze owner.

    O-1a-W1A: the accepted frozen configuration is the worker's only
    run-configuration authority, so a worker-level test states its intent here
    rather than on the mutable display scan.  Nothing is faked: ``RunIntent`` and
    ``GIIntent`` are the production Controls value types and ``freeze()`` is the
    production freeze.
    """
    from xrd_tools.session import RunIntent
    from xrd_tools.session.run_configuration import GIIntent

    return RunIntent(
        processing_mode="Int 1D" if skip_2d else "Int 2D",
        # O-1a-W1R (review §39.2 W1R-P1-4): the worker consumes the output mode
        # and the live/batch mode from the ACCEPTED object, so a case that wants
        # Overwrite/Replace or live watch must FREEZE it rather than set the (now
        # zero-reader) thread mirror.
        output_mode=output_mode,
        live_mode=live_mode,
        bai_1d_args=dict(bai_1d_args if bai_1d_args is not None
                         else {"unit": "q_A^-1"}),
        bai_2d_args=dict(bai_2d_args or {}),
        gi=GIIntent(enabled=bool(gi), mode_1d=gi_mode_1d, mode_2d=gi_mode_2d),
    ).freeze()


def _bare_worker(tmp_path, *, write_mode="Append"):
    worker = imageThread.__new__(imageThread)
    worker.write_mode = write_mode
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
    # the mutable DISPLAY scan double: retained so the backward GI projection has
    # a target, deliberately NOT the run-configuration source any more.
    worker.scan = SimpleNamespace(
        skip_2d=True,
        gi=False,
        bai_1d_args={"unit": "q_A^-1"},
        bai_2d_args={},
        gi_config={},
    )
    worker.run_configuration = worker._admitted_run_configuration = (
        _frozen_run_config(output_mode=write_mode))
    worker.run_configuration_floor = 0
    return worker


def _write_minimal_integrated_nxs(
        path, labels, *, labels_2d=None, reduction_config=None,
        source_path=None, source_snapshot=None):
    import h5py
    from xrd_tools.core.provenance import write_provenance
    from xrd_tools.io.nexus_record import write_frame_source_ref

    labels = np.asarray(labels, dtype=np.int64)
    q = np.linspace(0.1, 1.0, 4, dtype=np.float32)
    with h5py.File(path, "w") as h5:
        entry = h5.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        g1 = entry.create_group("integrated_1d")
        g1.attrs["NX_class"] = "NXdata"
        g1.attrs["signal"] = "intensity"
        g1.attrs["axes"] = ["frame_index", "q"]
        g1.create_dataset("frame_index", data=labels)
        q_ds = g1.create_dataset("q", data=q)
        q_ds.attrs["units"] = "q_A^-1"
        g1.create_dataset(
            "intensity",
            data=np.arange(labels.size * q.size, dtype=np.float32).reshape(
                labels.size, q.size
            ),
        )
        if labels_2d is not None:
            g2 = entry.create_group("integrated_2d")
            g2.create_dataset(
                "frame_index", data=np.asarray(labels_2d, dtype=np.int64))
        if source_path is not None and labels.size:
            frames = entry.create_group("frames")
            frame = frames.create_group(
                f"frame_{int(labels.max()):04d}")
            write_frame_source_ref(
                frame,
                source_path,
                int(labels.max()) - 1,
                source_snapshot=source_snapshot,
            )
        if reduction_config is not None:
            write_provenance(h5, config=reduction_config, host="")


def _initialize_scan_worker(tmp_path, *, write_mode="Append"):
    worker = _bare_worker(tmp_path, write_mode=write_mode)
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    worker.h5_dir = str(out)
    worker.scan_name = "scan"
    worker.scan = SimpleNamespace(
        skip_2d=True,
        gi=False,
        bai_1d_args={"unit": "q_A^-1"},
        bai_2d_args={},
        gi_config={},
    )
    worker.run_configuration = worker._admitted_run_configuration = (
        _frozen_run_config(output_mode=write_mode))
    worker.run_configuration_floor = 0
    worker.sigUpdateFile = SimpleNamespace(emit=lambda *_: None)

    @contextmanager
    def _noop_h5pool_bracket(_scan):
        yield

    worker._h5pool_bracket = _noop_h5pool_bracket
    return worker, out / "scan.nxs"


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


def test_image_series_selected_member_does_not_truncate_full_series(
    monkeypatch, tmp_path,
):
    """The selected member is context, not a lower frame bound."""
    paths = [
        tmp_path / "scan_1.tif",
        tmp_path / "scan-2.tif",
        tmp_path / "scan_10.tif",
    ]
    for path in paths:
        path.touch()

    worker = _bare_worker(tmp_path)
    worker.inp_type = "Image Series"
    worker.img_file = str(paths[1])
    worker.img_dir = str(tmp_path)
    worker.img_ext = "tif"
    worker.scan_name = "scan"
    worker.img_fnames = []
    worker.processed = []
    from xrd_tools.sources import image_series_spec
    worker.source_spec = image_series_spec(paths[1])

    read_calls = []

    def fake_read(path):
        read_calls.append(Path(path).name)
        return np.ones((2, 2), dtype=float)

    monkeypatch.setattr(iwt, "read_image", fake_read)

    observed = [worker.get_next_image() for _ in paths]

    assert [(Path(item[0]).name, item[2]) for item in observed] == [
        ("scan_1.tif", 1),
        ("scan-2.tif", 2),
        ("scan_10.tif", 10),
    ]
    assert read_calls == ["scan_1.tif", "scan-2.tif", "scan_10.tif"]


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


def test_append_skip_snapshot_lazily_reads_only_current_frame_index(
        monkeypatch, tmp_path):
    worker = _bare_worker(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    output = out / "scan.nxs"
    _write_minimal_integrated_nxs(output, [1, 2])

    def fail_live_scan(*_args, **_kwargs):
        pytest.fail("append cursor must not hydrate a LiveScan")

    monkeypatch.setattr(iwt, "LiveScan", fail_live_scan)

    assert worker._should_skip_before_read("scan", 1) is True
    assert worker._append_skip_frames_by_scan == {"scan": {1, 2}}


def test_int_2d_append_does_not_skip_frame_with_only_1d_output(tmp_path):
    worker = _bare_worker(tmp_path)
    worker.run_configuration = worker._admitted_run_configuration = _frozen_run_config(skip_2d=False)
    out = tmp_path / "out"
    out.mkdir()
    _write_minimal_integrated_nxs(out / "scan.nxs", [0])

    assert worker._should_skip_before_read("scan", 0) is False
    assert worker._append_skip_frames_by_scan == {"scan": set()}


def test_int_2d_append_completion_is_1d_2d_intersection(tmp_path):
    worker = _bare_worker(tmp_path)
    worker.run_configuration = worker._admitted_run_configuration = _frozen_run_config(skip_2d=False)
    out = tmp_path / "out"
    out.mkdir()
    _write_minimal_integrated_nxs(
        out / "scan.nxs", [1, 2], labels_2d=[2, 3])

    assert worker._should_skip_before_read("scan", 1) is False
    assert worker._should_skip_before_read("scan", 2) is True
    assert worker._should_skip_before_read("scan", 3) is False
    assert worker._append_skip_frames_by_scan == {"scan": {2}}


def test_int_1d_append_completion_requires_only_1d_output(tmp_path):
    worker = _bare_worker(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    _write_minimal_integrated_nxs(out / "scan.nxs", [0])

    assert worker._should_skip_before_read("scan", 0) is True


def test_append_dispatch_guard_uses_mode_aware_cursor_not_union_index(tmp_path):
    worker = _bare_worker(tmp_path)
    worker.run_configuration = worker._admitted_run_configuration = _frozen_run_config(skip_2d=False)
    worker._append_skip_frames_by_scan = {"scan": set()}
    loaded_scan = SimpleNamespace(frames=SimpleNamespace(index=[0]))

    assert worker._append_frame_complete("scan", 0, loaded_scan) is False


def test_append_cursor_rejects_reached_target_config_before_skip(tmp_path):
    from xrd_tools.session.readiness import AppendConfigMismatchError

    worker = _bare_worker(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    _write_minimal_integrated_nxs(
        out / "scan.nxs",
        [0],
        reduction_config={
            "gi": False,
            "bai_1d_args": {"unit": "q_A^-1", "numpoints": 1000},
            "bai_2d_args": {"unit": "q_A^-1"},
        },
    )
    worker.run_configuration = worker._admitted_run_configuration = _frozen_run_config(
        bai_1d_args={"unit": "q_A^-1", "numpoints": 500})

    with pytest.raises(AppendConfigMismatchError) as excinfo:
        worker._should_skip_before_read("scan", 0)

    message = str(excinfo.value)
    assert "Changed settings:\n- 1D points: existing 1000; current 500" in message
    assert "\n\nThe append target scan.nxs was preserved.\n\n" in message
    assert "Switch output mode to Replace" in message
    assert worker._append_skip_frames_by_scan == {}


def test_append_skip_snapshot_opens_each_reached_output_once(monkeypatch, tmp_path):
    worker = _bare_worker(tmp_path)
    worker.fname = str(tmp_path / "out" / "current.nxs")
    out = tmp_path / "out"
    out.mkdir()
    for name in ("scan_1", "scan_2", "scan_10"):
        (out / f"{name}.nxs").touch()

    calls = []

    def append_cursor(path, *, require_2d):
        calls.append(Path(path).name)
        return {1}, {}

    monkeypatch.setattr(iwt, "_nexus_append_cursor", append_cursor)

    assert worker._should_skip_before_read("scan_2", 1) is True
    assert worker._should_skip_before_read("scan_2", 1) is True
    assert calls == ["scan_2.nxs"]
    assert set(worker._append_skip_frames_by_scan) == {"scan_2"}
    assert Path(worker.fname).name == "current.nxs"


def test_append_cursor_memo_reuses_only_unchanged_output_across_runs(
        monkeypatch, tmp_path):
    """Stop -> Run reuses a stamp-qualified processed-output cursor."""
    worker = _bare_worker(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    output = out / "scan.nxs"
    _write_minimal_integrated_nxs(output, [1, 2])
    real_cursor = iwt._nexus_append_cursor

    assert worker._load_append_skip_snapshot("scan") == {1, 2}
    assert worker._append_cursor_memo

    worker._append_skip_frames_by_scan = {}
    monkeypatch.setattr(
        iwt,
        "_nexus_append_cursor",
        lambda *_args, **_kwargs: pytest.fail(
            "unchanged processed output was reopened"),
    )
    assert worker._load_append_skip_snapshot("scan") == {1, 2}

    # Replacing the product changes its stamp and must invalidate the memo.
    _write_minimal_integrated_nxs(output, [1, 2, 3])
    calls = []

    def cursor(path, *, require_2d):
        calls.append(Path(path).name)
        return real_cursor(path, require_2d=require_2d)

    monkeypatch.setattr(iwt, "_nexus_append_cursor", cursor)
    worker._append_skip_frames_by_scan = {}
    assert worker._load_append_skip_snapshot("scan") == {1, 2, 3}
    assert calls == ["scan.nxs"]


def test_append_cursor_memo_revalidates_current_processing_config(
        monkeypatch, tmp_path):
    from xrd_tools.session.readiness import AppendConfigMismatchError

    worker = _bare_worker(tmp_path)
    worker.run_configuration = worker._admitted_run_configuration = _frozen_run_config(
        bai_1d_args={"unit": "q_A^-1", "numpoints": 1000})
    out = tmp_path / "out"
    out.mkdir()
    _write_minimal_integrated_nxs(
        out / "scan.nxs",
        [1, 2],
        reduction_config={
            "gi": False,
            "bai_1d_args": {"unit": "q_A^-1", "numpoints": 1000},
            "bai_2d_args": {"unit": "q_A^-1"},
        },
    )

    assert worker._load_append_skip_snapshot("scan") == {1, 2}
    worker._append_skip_frames_by_scan = {}
    worker.run_configuration = worker._admitted_run_configuration = _frozen_run_config(
        bai_1d_args={"unit": "q_A^-1", "numpoints": 500})
    monkeypatch.setattr(
        iwt,
        "_nexus_append_cursor",
        lambda *_args, **_kwargs: pytest.fail(
            "config revalidation reopened an unchanged output"),
    )

    with pytest.raises(AppendConfigMismatchError):
        worker._load_append_skip_snapshot("scan")


def test_append_cursor_memo_avoids_reopening_300_unchanged_products(
        monkeypatch, tmp_path):
    worker = _bare_worker(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    names = tuple(f"scan_{idx:04d}" for idx in range(300))
    for name in names:
        (out / f"{name}.nxs").write_bytes(name.encode())

    calls = []

    def cursor(path, *, require_2d):
        calls.append(Path(path).name)
        return set(range(1, 7)), {}

    monkeypatch.setattr(iwt, "_nexus_append_cursor", cursor)
    for name in names:
        assert worker._load_append_skip_snapshot(name) == set(range(1, 7))
    assert len(calls) == 300

    worker._append_skip_frames_by_scan = {}
    calls.clear()
    for name in names:
        assert worker._load_append_skip_snapshot(name) == set(range(1, 7))
    assert calls == []


def test_append_skip_snapshot_stop_does_not_open_output(monkeypatch, tmp_path):
    worker = _bare_worker(tmp_path)
    worker.command = "stop"

    monkeypatch.setattr(
        iwt,
        "_nexus_append_cursor",
        lambda *_args, **_kwargs: pytest.fail(
            "Stop must not start an append cursor read"),
    )

    assert worker._should_skip_before_read("scan", 1) is False
    assert worker._append_skip_frames_by_scan == {}


def test_stop_during_append_cursor_read_prevents_next_output_open(
        monkeypatch, tmp_path):
    worker = _bare_worker(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    for name in ("scan_1", "scan_2"):
        (out / f"{name}.nxs").touch()
    calls = []

    def first_cursor_then_stop(path, *, require_2d):
        calls.append(Path(path).name)
        worker.command = "stop"
        return {1}, {}

    monkeypatch.setattr(
        iwt, "_nexus_append_cursor", first_cursor_then_stop)

    assert worker._should_skip_before_read("scan_1", 1) is True
    assert worker._should_skip_before_read("scan_2", 1) is False
    assert calls == ["scan_1.nxs"]


def test_normal_append_process_start_never_primes_whole_directory(tmp_path):
    worker = _bare_worker(tmp_path)
    worker.batch_mode = True
    worker.live_mode = False
    worker._frames_since_save = 0
    worker.get_next_image = lambda: (None, None, 1, None, {})
    worker._wait_if_paused = lambda: None
    worker.flush_serial_tail = lambda *_args, **_kwargs: None
    worker._prime_append_skip_snapshots_for_run = lambda: pytest.fail(
        "ordinary Append startup performed an all-directory output sweep"
    )
    worker.sigUpdate = SimpleNamespace(emit=lambda *_: None)

    worker.process_scan()

    assert worker.files_processed == 0


def _current_candidate(path):
    import xrd_tools.sources.registry  # noqa: F401
    from xrd_tools.sources.adapters import candidate_owner
    from xrd_tools.sources.discover import Candidate

    stat = path.stat()
    owner = candidate_owner(path)
    assert owner is not None
    return Candidate(
        path, owner.id, int(stat.st_size), int(stat.st_mtime_ns))


def test_live_idle_recursive_discovery_is_bounded_per_reader_request(tmp_path):
    """An accidental broad root cannot be fully traversed in one live tick."""
    from xrd_tools.sources import DirectorySourceSpec

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for index in range(150):
        child = raw_dir / f"child_{index:03d}"
        child.mkdir()
        (child / "notes.txt").write_text("not a source")

    worker = _bare_worker(tmp_path)
    worker.inp_type = "Image Directory"
    worker.img_ext = "nxs"
    worker.img_dir = str(raw_dir)
    worker.include_subdir = True
    worker.source_spec = DirectorySourceSpec(
        raw_dir, recursive=True, suffixes=(".nxs",))
    worker.source_run_plan = None
    worker.source_index_session = None
    # O-1a-W1R: live/batch mode is FROZEN policy -- the bounded-discovery branch
    # in `_eiger_pop_next_master` reads it, so freeze it here.
    worker.run_configuration = worker._admitted_run_configuration = (
        _frozen_run_config(live_mode=True))
    worker.live_mode = True
    worker._eiger_master_path = None
    worker._eiger_frame_idx = 0
    worker._eiger_nframes = 0
    worker._eiger_master_queue = deque()
    worker._directory_walk_iter = None
    worker._eiger_done_masters = set()
    worker._eiger_retry_after = {}
    worker._eiger_zero_frame_seen = {}
    worker._eiger_master_candidate = None
    worker._eiger_open_state = None

    assert worker._get_next_eiger_frame_sync() == (
        None, None, 1, None, {})
    assert worker._directory_walk_iter is not None
    assert worker._eiger_done_masters == set()


def test_complete_append_master_uses_count_hint_without_raw_open(
        monkeypatch, tmp_path):
    """A restarted Append run retires a complete container as one unit."""
    from xrd_tools.sources.directory_index import Snapshot
    from xrd_tools.sources.run_plan import RunCandidatePlan

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw = raw_dir / "scan.nxs"
    raw.write_bytes(b"source identity only")
    candidate = _current_candidate(raw)

    worker = _bare_worker(tmp_path)
    worker.inp_type = "Image Directory"
    worker.img_ext = "nxs"
    worker.img_dir = str(raw_dir)
    worker.live_mode = False
    worker._eiger_master_path = None
    worker._eiger_frame_idx = 0
    worker._eiger_nframes = 0
    worker._eiger_master_queue = deque()
    worker._eiger_done_masters = set()
    worker._eiger_retry_after = {}
    worker._eiger_zero_frame_seen = {}
    worker._source_plan_reported = set()
    worker._h19_observed_master_paths = ()
    worker._h19_ready_master_candidates = {str(raw): candidate}
    worker._h19_seed_pending = True
    worker._eiger_master_candidate = None
    worker.source_run_plan = RunCandidatePlan.from_snapshot(Snapshot(
        1, (candidate,), raw_dir, False, None))
    worker.source_index_session = object()
    worker.source_frame_count_snapshot = {
        str(raw): (candidate.version_stamp, 6)}
    worker.showLabel = SimpleNamespace(emit=lambda *_: None)

    out = Path(worker.h5_dir)
    out.mkdir()
    _write_minimal_integrated_nxs(out / "scan.nxs", range(1, 7))
    monkeypatch.setattr(
        worker,
        "_eiger_open_master",
        lambda _path: pytest.fail("complete raw master was opened"),
    )

    result = worker._get_next_eiger_frame_sync()

    assert result == (None, None, 1, None, {})
    assert worker._eiger_done_masters == {str(raw)}
    assert worker._append_skip_without_reading == 6
    assert worker._discovered_frame_count == 6
    assert worker._skip_reason_counts == Counter({"already processed": 6})


def test_restarted_append_uses_persisted_source_stamp_without_raw_open(
        monkeypatch, tmp_path):
    """A new worker can validate a completed self-contained source by stat.

    The optimization evidence lives in the processed product, not only in a
    same-process GUI memo.  Directory enumeration remains lazy and the raw
    container is never opened when its exact stamp and full extent match.
    """
    from xrd_tools.sources import DirectorySourceSpec

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw = raw_dir / "scan.nxs"
    raw.write_bytes(b"stable self-contained source identity")
    candidate = _current_candidate(raw)

    worker = _bare_worker(tmp_path)
    worker.inp_type = "Image Directory"
    worker.img_ext = "nxs"
    worker.img_dir = str(raw_dir)
    worker.source_spec = DirectorySourceSpec(
        raw_dir, suffixes=(".nxs",))
    worker.source_run_plan = None
    worker.source_index_session = None
    worker.source_frame_count_snapshot = {}
    worker.live_mode = False
    worker._eiger_master_path = None
    worker._eiger_frame_idx = 0
    worker._eiger_nframes = 0
    worker._eiger_master_queue = deque()
    worker._directory_walk_iter = None
    worker._eiger_done_masters = set()
    worker._eiger_retry_after = {}
    worker._eiger_zero_frame_seen = {}
    worker._eiger_master_candidate = None
    worker._eiger_open_state = None
    worker._eiger_cursor = None
    worker._eiger_descriptor = None
    worker._eiger_read_plan = None
    worker._eiger_provider = None
    worker._eiger_fabio_handle = None
    worker._source_snapshot_by_path = {}

    out = Path(worker.h5_dir)
    out.mkdir()
    _write_minimal_integrated_nxs(
        out / "scan.nxs",
        range(1, 7),
        source_path=raw,
        source_snapshot={
            "size": candidate.size,
            "mtime_ns": candidate.mtime_ns,
            "frame_count": 6,
            "dataset_path": "/entry/data/detector",
            "self_contained": True,
        },
    )
    monkeypatch.setattr(
        worker,
        "_eiger_open_master",
        lambda _path: pytest.fail(
            "unchanged complete source was opened"),
    )

    assert worker._get_next_eiger_frame_sync() == (
        None, None, 1, None, {})
    assert worker._eiger_done_masters == {str(raw)}
    assert worker._append_skip_without_reading == 6
    assert worker._append_source_snapshot_by_scan["scan"][
        "frame_count"] == 6


def test_changed_source_stamp_invalidates_persisted_append_extent(tmp_path):
    """A growing/replaced source never inherits an old complete decision."""
    raw = tmp_path / "scan.nxs"
    raw.write_bytes(b"old")
    old = _current_candidate(raw)

    worker = _bare_worker(tmp_path)
    worker._eiger_done_masters = set()
    out = Path(worker.h5_dir)
    out.mkdir()
    _write_minimal_integrated_nxs(
        out / "scan.nxs",
        range(1, 7),
        source_path=raw,
        source_snapshot={
            "size": old.size,
            "mtime_ns": old.mtime_ns,
            "frame_count": 6,
            "self_contained": True,
        },
    )

    raw.write_bytes(b"new source extent is larger")
    current = _current_candidate(raw)
    assert current.version_stamp != old.version_stamp
    assert worker._eiger_skip_complete_append_master(
        str(raw), current) is False
    assert worker._eiger_done_masters == set()


def test_eiger_master_never_uses_persisted_mtime_as_completeness(tmp_path):
    """External Eiger data can grow while the master stamp stays unchanged."""
    raw = tmp_path / "scan_master.h5"
    raw.write_bytes(b"stable master bytes, external data may grow")
    candidate = _current_candidate(raw)

    worker = _bare_worker(tmp_path)
    worker._eiger_done_masters = set()
    out = Path(worker.h5_dir)
    out.mkdir()
    _write_minimal_integrated_nxs(
        out / "scan.nxs",
        range(1, 7),
        source_path=raw,
        source_snapshot={
            "size": candidate.size,
            "mtime_ns": candidate.mtime_ns,
            "frame_count": 6,
            "self_contained": False,
        },
    )

    assert worker._eiger_skip_complete_append_master(
        str(raw), candidate) is False
    assert worker._eiger_done_masters == set()


def test_cold_append_opens_complete_master_once_without_frame_index_walk(
        monkeypatch, tmp_path):
    """A fresh GUI uses the first raw open's count and skips as one unit."""
    import h5py
    from xrd_tools.sources.cursor import ContainerCursor
    from xrd_tools.sources.directory_index import Snapshot
    from xrd_tools.sources.run_plan import RunCandidatePlan

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw = raw_dir / "scan.nxs"
    with h5py.File(raw, "w") as handle:
        detector = handle.create_group("entry/instrument/detector")
        detector.create_dataset(
            "data", data=np.zeros((6, 3, 4), dtype=np.uint16))
    candidate = _current_candidate(raw)

    worker = _bare_worker(tmp_path)
    worker.inp_type = "Image Directory"
    worker.img_ext = "nxs"
    worker.img_dir = str(raw_dir)
    worker.live_mode = False
    worker._eiger_master_path = None
    worker._eiger_frame_idx = 0
    worker._eiger_nframes = 0
    worker._eiger_master_queue = deque()
    worker._eiger_done_masters = set()
    worker._eiger_retry_after = {}
    worker._eiger_zero_frame_seen = {}
    worker._source_plan_reported = set()
    worker._h19_observed_master_paths = ()
    worker._h19_ready_master_candidates = {str(raw): candidate}
    worker._h19_seed_pending = True
    worker._h19_pending_count = 0
    worker._eiger_master_candidate = None
    worker._eiger_cursor = None
    worker._eiger_descriptor = None
    worker._eiger_read_plan = None
    worker._eiger_provider = None
    worker._eiger_fabio_handle = None
    worker.source_run_plan = RunCandidatePlan.from_snapshot(Snapshot(
        1, (candidate,), raw_dir, False, None))
    worker.source_index_session = object()
    worker.source_frame_count_snapshot = {}

    out = Path(worker.h5_dir)
    out.mkdir()
    _write_minimal_integrated_nxs(out / "scan.nxs", range(1, 7))

    opens = []

    def open_once(path):
        opens.append(str(path))
        cursor = ContainerCursor(path, candidate=candidate).open()
        worker._eiger_cursor = cursor
        worker._eiger_descriptor = cursor.descriptor
        worker._eiger_nframes = cursor.frame_count

    monkeypatch.setattr(worker, "_eiger_open_master", open_once)
    monkeypatch.setattr(
        worker,
        "_should_skip_before_read",
        lambda *_args: pytest.fail("complete container walked frame indices"),
    )

    result = worker._get_next_eiger_frame_sync()

    assert result == (None, None, 1, None, {})
    assert opens == [str(raw)]
    assert worker._eiger_done_masters == {str(raw)}
    assert worker._append_skip_without_reading == 6
    assert worker._discovered_frame_count == 6
    assert worker._skip_reason_counts == Counter({"already processed": 6})
    assert worker._eiger_cursor is None


def test_cold_append_tail_seek_accounts_completed_prefix(tmp_path):
    raw = tmp_path / "scan.nxs"
    raw.touch()
    worker = _bare_worker(tmp_path)
    worker.inp_type = "Image Directory"
    worker._eiger_master_path = str(raw)
    worker._eiger_nframes = 7
    worker._eiger_frame_idx = 0
    worker._eiger_cursor = object()
    worker._eiger_descriptor = object()
    worker._append_skip_frames_by_scan = {
        "scan": set(range(1, 6))}

    assert worker._eiger_skip_open_complete_append_master() is False
    assert worker._eiger_frame_idx == 5
    assert worker._append_skip_without_reading == 5
    assert worker._discovered_frame_count == 5
    assert worker._skip_reason_counts == Counter(
        {"already processed": 5})


def test_external_nexus_master_reopens_then_seeks_missing_tail(tmp_path):
    """External masters use their current open extent, never persisted mtime."""
    import h5py

    target = tmp_path / "scan_data_000001.h5"
    with h5py.File(target, "w") as handle:
        handle.create_dataset(
            "frames", data=np.zeros((7, 3, 4), dtype=np.uint16))
    master = tmp_path / "scan.nxs"
    with h5py.File(master, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        data = entry.create_group("data")
        data["data_000001"] = h5py.ExternalLink(
            target.name, "/frames")

    worker = _bare_worker(tmp_path)
    worker.inp_type = "Image Directory"
    worker._eiger_master_path = str(master)
    worker._eiger_master_candidate = _current_candidate(master)
    worker._eiger_frame_idx = 0
    worker._eiger_nframes = 0
    worker._eiger_cursor = None
    worker._eiger_descriptor = None
    worker._eiger_read_plan = None
    worker._eiger_provider = None
    worker._eiger_fabio_handle = None
    worker._eiger_metadata_cache = {}
    worker._bluesky_source_cache = {}
    worker.sigContainerCount = None
    worker._append_skip_frames_by_scan = {
        "scan": set(range(1, 6))}

    worker._eiger_open_master(str(master))
    try:
        assert worker._eiger_nframes == 7
        assert worker._eiger_descriptor.self_contained is False
        assert worker._eiger_open_count_can_bulk_compare(str(master)) is True
        assert worker._eiger_skip_open_complete_append_master() is False
        assert worker._eiger_frame_idx == 5
        assert worker._append_skip_without_reading == 5
    finally:
        worker._eiger_close_master()


def test_complete_append_directory_retires_1800_frames_without_raw_open(
        monkeypatch, tmp_path):
    """A large restarted directory is skipped in container-sized units.

    This mirrors the beamline case (roughly 300 six-frame NeXus files): the
    worker may point-check each container identity and validate each processed
    output, but it must not reopen a raw container, iterate the frame reader, or
    re-walk the recursive directory before returning the no-op result.
    """
    from xrd_tools.sources.directory_index import Snapshot
    from xrd_tools.sources.run_plan import RunCandidatePlan

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    paths = tuple(raw_dir / f"scan_{idx:04d}.nxs" for idx in range(300))
    for path in paths:
        path.write_bytes(path.name.encode())
    candidates = tuple(_current_candidate(path) for path in paths)

    class NoObservation:
        def observe(self):
            pytest.fail("frozen READY queue was re-observed")

    worker = _bare_worker(tmp_path)
    worker.inp_type = "Image Directory"
    worker.img_ext = "nxs"
    worker.img_dir = str(raw_dir)
    worker.live_mode = False
    worker._eiger_master_path = None
    worker._eiger_frame_idx = 0
    worker._eiger_nframes = 0
    worker._eiger_master_queue = deque()
    worker._eiger_done_masters = set()
    worker._eiger_retry_after = {}
    worker._eiger_zero_frame_seen = {}
    worker._source_plan_reported = set()
    worker._h19_observed_master_paths = ()
    worker._h19_ready_master_candidates = {
        str(candidate.path): candidate for candidate in candidates}
    worker._h19_seed_pending = True
    worker._eiger_master_candidate = None
    worker.source_run_plan = RunCandidatePlan.from_snapshot(Snapshot(
        1, candidates, raw_dir, True, None))
    worker.source_index_session = NoObservation()
    worker.source_frame_count_snapshot = {
        str(candidate.path): (candidate.version_stamp, 6)
        for candidate in candidates
    }
    worker._append_skip_frames_by_scan = {
        worker._eiger_scan_name(candidate.path): set(range(1, 7))
        for candidate in candidates
    }
    monkeypatch.setattr(
        worker,
        "_eiger_open_master",
        lambda _path: pytest.fail("complete raw container was opened"),
    )

    result = worker._get_next_eiger_frame_sync()

    assert result == (None, None, 1, None, {})
    assert worker._eiger_done_masters == {str(path) for path in paths}
    assert worker._append_skip_without_reading == 1800
    assert worker._discovered_frame_count == 1800
    assert worker._skip_reason_counts == Counter(
        {"already processed": 1800})


@pytest.mark.parametrize("case", ("stale-count", "partial-output"))
def test_append_master_count_hint_falls_back_when_not_authoritative(
        case, tmp_path):
    raw = tmp_path / "scan.nxs"
    raw.write_bytes(b"source identity only")
    candidate = _current_candidate(raw)
    worker = _bare_worker(tmp_path)
    worker._eiger_done_masters = set()
    worker.source_frame_count_snapshot = {
        str(raw): (
            (candidate.size + 1, candidate.mtime_ns)
            if case == "stale-count" else candidate.version_stamp,
            6,
        )
    }
    out = Path(worker.h5_dir)
    out.mkdir()
    labels = range(1, 7) if case == "stale-count" else range(1, 6)
    _write_minimal_integrated_nxs(out / "scan.nxs", labels)

    assert worker._eiger_skip_complete_append_master(
        str(raw), candidate) is False
    assert worker._eiger_done_masters == set()
    assert worker._append_skip_without_reading == 0


def test_append_skip_snapshot_primes_once_read_only(monkeypatch, tmp_path):
    worker = _bare_worker(tmp_path)
    worker.inp_type = "Image File"
    worker.img_file = str(tmp_path / "scan_master.h5")
    worker.img_ext = "h5"
    worker.scan_name = "scan_master"
    out = tmp_path / "out"
    out.mkdir()
    output = out / "scan.nxs"
    _write_minimal_integrated_nxs(output, [1, 2])
    calls = []

    def append_cursor(path, *, require_2d):
        calls.append(str(path))
        return {1, 2}, {}

    monkeypatch.setattr(iwt, "_nexus_append_cursor", append_cursor)

    worker._prime_append_skip_snapshots_for_run()
    worker._prime_append_skip_snapshots_for_run()

    assert calls == [str(output)]
    assert worker._append_skip_frames_by_scan == {"scan": {1, 2}}
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
    _write_minimal_integrated_nxs(output, [4, 5])

    with h5py.File(output, "r"):
        worker._prime_append_skip_snapshots_for_run()

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

    def fail_snapshot(*_args, **_kwargs):
        pytest.fail("fresh append snapshot should not open a missing .nxs")

    read_calls = []

    def fake_read(path):
        read_calls.append(os.fspath(path))
        return np.ones((2, 2), dtype=float)

    monkeypatch.setattr(iwt, "_nexus_append_cursor", fail_snapshot)
    monkeypatch.setattr(iwt, "read_image", fake_read)

    worker._prime_append_skip_snapshots_for_run()
    img_file, scan_name, img_number, img_data, _meta = worker.get_next_image()

    assert worker._append_skip_frames_by_scan == {"scan": set()}
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

    monkeypatch.setattr(
        iwt,
        "_nexus_append_cursor",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("held read handle")),
    )

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


def test_nexus_integrated_frame_labels_unions_1d_and_2d(tmp_path):
    import h5py

    output = tmp_path / "scan.nxs"
    with h5py.File(output, "w") as h5:
        entry = h5.create_group("entry")
        entry.create_group("integrated_1d").create_dataset(
            "frame_index", data=np.asarray([1, 3], dtype=np.int64))
        entry.create_group("integrated_2d").create_dataset(
            "frame_index", data=np.asarray([2, 3], dtype=np.int64))

    assert iwt._nexus_integrated_frame_labels(output) == {1, 2, 3}
    assert iwt._nexus_integrated_frame_count(output) == 3


def test_append_initialize_abort_preserves_target_on_degraded_load(
        monkeypatch, tmp_path):
    import xrd_tools.io.nexus as nexus_mod

    worker, target = _initialize_scan_worker(tmp_path)
    _write_minimal_integrated_nxs(target, [1, 2, 3])
    before = target.read_bytes()

    def fail_read_scan_metadata(*_args, **_kwargs):
        raise OSError("transient metadata read failure")

    monkeypatch.setattr(
        nexus_mod, "read_scan_metadata", fail_read_scan_metadata)

    with pytest.raises(RuntimeError, match="existing file preserved"):
        worker.initialize_scan()

    assert target.read_bytes() == before


def test_append_initialize_marks_loaded_rows_persisted(tmp_path):
    from xdart.gui.tabs.static_scan.wranglers.qt_nexus_sink import QtNexusSink

    worker, target = _initialize_scan_worker(tmp_path)
    _write_minimal_integrated_nxs(target, [1, 2, 3])

    scan = worker.initialize_scan()

    assert scan.frames.index == [1, 2, 3]
    assert set(scan.frames._persisted) == {1, 2, 3}
    sink = QtNexusSink(SimpleNamespace(batch_mode=True), scan, object())
    assert sink._needs_atomic_first_batch_flush() is False


def test_append_initialize_config_mismatch_aborts_with_empty_img_file(tmp_path):
    worker, target = _initialize_scan_worker(tmp_path)
    _write_minimal_integrated_nxs(
        target,
        [1, 2, 3],
        reduction_config={
            "gi": False,
            "bai_1d_args": {"unit": "q_A^-1"},
            "bai_2d_args": {"unit": "q_A^-1"},
        },
    )
    before = target.read_bytes()
    worker.img_file = ""  # live/cold path: Run click may not know the source yet.
    worker.gi = True
    worker.run_configuration = worker._admitted_run_configuration = _frozen_run_config(
        gi=True,
        bai_1d_args={"unit": "q_A^-1", "gi_mode_1d": "q_total"},
        bai_2d_args={"unit": "q_A^-1", "gi_mode_2d": "qip_qoop"},
    )

    with pytest.raises(RuntimeError, match="Integration settings changed mid-run") as excinfo:
        worker.initialize_scan()

    # TYPED (still a RuntimeError subclass) so run()/process_scan can stop the
    # run cleanly instead of crashing the QThread; carries the check and names
    # the differing fields + the preserved target in the user-facing message.
    from xrd_tools.session.readiness import AppendConfigMismatchError

    err = excinfo.value
    assert isinstance(err, AppendConfigMismatchError)
    assert err.check.ok is False
    assert err.check.mismatched_fields
    assert (err.check.processed_label, err.check.current_label) == (
        "Standard", "Grazing")
    for label in err.check.mismatched_fields:
        assert label in str(err)
    assert "scan.nxs was preserved" in str(err)
    assert target.read_bytes() == before


def test_append_initialize_same_config_canonical_noise_passes(tmp_path):
    worker, target = _initialize_scan_worker(tmp_path)
    _write_minimal_integrated_nxs(
        target,
        [1, 2, 3],
        reduction_config={
            "gi": "False",
            "bai_1d_args": {
                "unit": " q_A^-1 ",
                "numpoints": None,
                "radial_range": "[0.1, 6.0]",
            },
            "bai_2d_args": {
                "unit": "q_A^-1",
                "npt_rad": "500.0",
                "npt_azim": "",
                "radial_range": ["0.1000000000001", "6.0"],
                "azimuth_range": "(-90.0, 90.0)",
            },
        },
    )
    worker.img_file = ""
    worker.run_configuration = worker._admitted_run_configuration = _frozen_run_config(
        bai_1d_args={
            "unit": "q_A^-1",
            "radial_range": (0.1, 6.0),
        },
        bai_2d_args={
            "unit": "q_A^-1",
            "npt_rad": 500,
            "radial_range": (0.1, 6.0),
            "azimuth_range": (-90, 90),
        },
    )

    scan = worker.initialize_scan()

    assert scan.frames.index == [1, 2, 3]


def test_append_initialize_mismatch_ignored_in_overwrite_mode(tmp_path):
    worker, target = _initialize_scan_worker(tmp_path, write_mode="Overwrite")
    _write_minimal_integrated_nxs(
        target,
        [1, 2, 3],
        reduction_config={
            "gi": False,
            "bai_1d_args": {"unit": "q_A^-1"},
            "bai_2d_args": {"unit": "q_A^-1"},
        },
    )
    worker.img_file = ""
    worker.gi = True
    worker.run_configuration = worker._admitted_run_configuration = _frozen_run_config(
        output_mode="Overwrite",
        gi=True,
        bai_1d_args={"unit": "q_A^-1", "gi_mode_1d": "q_total"},
        bai_2d_args={"unit": "q_A^-1", "gi_mode_2d": "qip_qoop"},
    )

    scan = worker.initialize_scan()

    assert scan.frames.index == []


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


def test_zero_processed_no_discovered_frames_warns_with_source_details(
        caplog, tmp_path):
    worker = _bare_worker(tmp_path)
    labels = []
    worker.showLabel = SimpleNamespace(emit=labels.append)
    worker.img_dir = str(tmp_path / "empty")
    worker.img_ext = "TIF"
    worker.file_filter = "scan | sample"

    with caplog.at_level(logging.WARNING):
        worker._report_run_skip_summary(0)

    assert labels
    message = labels[0]
    assert "No frames discovered" in message
    assert f"directory: {tmp_path / 'empty'}" in message
    assert "ext: .TIF" in message
    assert "pattern: scan | sample" in message
    assert message in caplog.text


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
    # O-1a-W1R: the output mode is FROZEN policy, so a case that wants Replace
    # must freeze it -- the thread mirror has no execution readers left.
    worker = _bare_worker(tmp_path, write_mode="Overwrite")
    worker.series_average = True
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
