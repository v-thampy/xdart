"""Production-wired NXS directory input/output safety (v1.1.2 Tranche A).

Drives the REAL ``imageThread`` (full ``__init__``) — no fakes on the seam being
fixed — to pin two reproduced production-path failures:

* **F-NXS-1** — Save Path equal to the raw source directory made source == output
  for a same-stem container, and ``initialize_scan`` Overwrite-saved it,
  replacing a real raw acquisition with an empty processed container before any
  frame was reduced.  The run must be refused BEFORE any write, with the source
  bytes/size/tree intact.
* **F-NXS-2** — a processed xdart ``.nxs`` swept into a watched raw directory was
  consumed as raw input (its ``integrated_2d`` cake returned as detector frames).
  The directory reader must skip it per file and continue to the next valid raw
  acquisition.

The fail-before behaviour (against the unfixed tree) is recorded in the tranche
report; these tests assert the fixed behaviour.
"""
from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from queue import Queue

import h5py
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.core.test_bluesky_nexus import _write_bluesky_nxwriter  # noqa: E402
from tests.core.test_output_safety import _write_processed_xdart  # noqa: E402

from xrd_tools.core.containers import PONI  # noqa: E402
from xrd_tools.io.output_safety import (  # noqa: E402
    OutputCollisionError,
    check_output_not_source,
)
from xdart.modules.live import LiveScan  # noqa: E402
from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import (  # noqa: E402
    imageThread,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

# Real imageThreads open persistent h5py handles and (via process_scan) spawn a
# background prefetch thread.  Track them and tear them down after each test —
# BEFORE tmp_path is deleted (the fixture depends on tmp_path, so it finalizes
# first) — so no dead thread touches a removed file (an h5py-in-dead-thread
# segfault at interpreter/GC time).
_LIVE_THREADS: list = []


@pytest.fixture(autouse=True)
def _teardown_live_threads(tmp_path):
    yield
    for t in _LIVE_THREADS:
        try:
            t.command = "stop"
            t._prefetch_stop_prior()
            t._eiger_close_master()
        except Exception:
            pass
    _LIVE_THREADS.clear()


def _make_thread(watch_dir, out_dir, *, img_ext="nxs", scan_name="scan",
                 include_subdir=False, img_file="", live_mode=False):
    """A REAL imageThread watching *watch_dir*, writing into *out_dir*."""
    scan = LiveScan("scan", data_file=str(Path(out_dir) / "scan.nxs"),
                    static=True)
    t = imageThread(
        Queue(), {}, threading.RLock(), "",
        str(out_dir),                # h5_dir
        scan_name,                   # scan_name
        False,                       # single_img
        PONI(dist=0.2, poni1=0.1, poni2=0.1, wavelength=1e-10),
        "Image Directory",           # inp_type
        str(img_file),               # img_file
        str(watch_dir),              # img_dir
        include_subdir,              # include_subdir
        img_ext,                     # img_ext
        False,                       # series_average
        None,                        # meta_ext
        "",                          # file_filter
        None,                        # mask_file
        "Full",                      # write_mode
        "None",                      # bg_type
        "", "", None, "", "",        # bg_*
        1.0, None,                   # bg_scale, bg_norm_channel
        False, None, 1, 0.0,         # gi, th_mtr, sample_orientation, tilt
        "q_total", "qip_qoop",       # gi modes
        "start", scan,
        live_mode=live_mode, max_cores=1,
    )
    _LIVE_THREADS.append(t)
    return t


def _hdf5_tree(path):
    sig = []
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                sig.append((name, tuple(obj.shape), str(obj.dtype)))
        f.visititems(visit)
    return sorted(sig)


def _digest(path):
    data = Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest(), len(data), _hdf5_tree(path)


def _drain_reader(t, limit=64):
    """Pull frames from the REAL directory reader until the stream ends."""
    frames = []
    for _ in range(limit):
        item = t._get_next_eiger_frame_sync()
        if item[3] is None:
            break
        # (scan_name, img_number, frame_shape)
        frames.append((item[1], item[2], np.asarray(item[3]).shape))
    return frames


# ---------------------------------------------------------------------------
# F-NXS-1 — raw source overwrite
# ---------------------------------------------------------------------------

def test_raw_source_preserved_when_save_path_equals_source_dir(tmp_path):
    """Save Path == the watched raw container directory: the run is refused and
    the raw acquisition's bytes, size and HDF5 tree are untouched."""
    src_dir = tmp_path / "raw"
    src_dir.mkdir()
    raw = src_dir / "LaB6_00007.nxs"
    _write_bluesky_nxwriter(raw, n=3)          # a real detector container
    before = _digest(raw)

    # h5_dir == img_dir == src_dir — the destructive configuration.
    t = _make_thread(src_dir, src_dir, scan_name="LaB6_00007")

    # THE writer seam: initialize_scan performs the destructive replace-save.
    with pytest.raises(OutputCollisionError):
        t.initialize_scan()

    # And the real run body refuses BEFORE reading/writing anything.
    t.command = "start"
    t.process_scan()
    assert t.command == "stop"

    after = _digest(raw)
    assert after == before, "raw source must be byte/size/tree-identical"


def test_recursive_output_under_watched_tree_rejected(tmp_path):
    """A recursive watch with the Save Path inside the watched tree is refused
    (the output would be re-discovered as a raw input)."""
    watch = tmp_path / "raw"
    out = watch / "processed"
    watch.mkdir()
    out.mkdir()
    raw = watch / "acq_00001.nxs"
    _write_bluesky_nxwriter(raw, n=2)
    before = _digest(raw)

    t = _make_thread(watch, out, scan_name="acq_00001", include_subdir=True)
    with pytest.raises(OutputCollisionError):
        t.initialize_scan()

    assert _digest(raw) == before


def test_separate_directories_process_normally(tmp_path):
    """Distinct input/output directories are NOT refused, and frames flow."""
    watch = tmp_path / "raw"
    out = tmp_path / "out"
    watch.mkdir()
    out.mkdir()
    _write_bluesky_nxwriter(watch / "acq_00001.nxs", n=3)

    t = _make_thread(watch, out, scan_name="acq_00001")
    # The safety owner (assembled from the REAL thread config) does not fire.
    check_output_not_source(str(out / "acq_00001.nxs"), **t._output_safety_args())
    # And the real reader yields the raw frames.
    frames = _drain_reader(t)
    assert [f[0] for f in frames] == ["acq_00001"] * 3
    assert [f[1] for f in frames] == [1, 2, 3]


# ---------------------------------------------------------------------------
# F-NXS-2 — processed-output re-ingestion
# ---------------------------------------------------------------------------

def test_processed_file_in_watched_dir_skipped_only_raw_yielded(tmp_path):
    """A processed xdart .nxs sorting FIRST in a watched directory is skipped
    per file; only the raw acquisition's detector frames are yielded (never the
    integrated_2d cake)."""
    watch = tmp_path / "watch"
    out = tmp_path / "out"
    watch.mkdir()
    out.mkdir()
    # Sorts first — the dangerous ordering (an unhandled processed file first
    # used to hand back its cake as detector frames).
    proc = watch / "00_previous_run_00001.nxs"
    _write_processed_xdart(proc)
    _write_bluesky_nxwriter(watch / "10_acq_00001.nxs", n=3)

    t = _make_thread(watch, out)
    frames = _drain_reader(t)

    assert {f[0] for f in frames} == {"10_acq_00001"}, \
        "processed output must not yield frames"
    assert [f[1] for f in frames] == [1, 2, 3]
    # The processed file was skip-and-retired, and the reason is recorded.
    assert str(proc) in t._eiger_done_masters
    assert t._skip_reason_counts.get("processed xdart output", 0) >= 1


def test_processed_and_imageless_mixed_skip_raw_flows(tmp_path):
    """Imageless AND processed containers interleaved with real data: every
    non-raw container is skipped, every raw frame still flows in order."""
    watch = tmp_path / "watch"
    out = tmp_path / "out"
    watch.mkdir()
    out.mkdir()
    # imageless alignment scan (no image dataset)
    with h5py.File(watch / "01_align_00001.nxs", "w") as f:
        e = f.create_group("entry")
        e.attrs["NX_class"] = "NXentry"
        d = e.create_group("data")
        d.create_dataset("i0", data=np.linspace(0.0, 1.0, 6))
    # processed output
    _write_processed_xdart(watch / "02_processed_00001.nxs")
    # two real data scans
    _write_bluesky_nxwriter(watch / "03_data_00001.nxs", n=2)
    _write_bluesky_nxwriter(watch / "04_data_00002.nxs", n=3)

    t = _make_thread(watch, out)
    frames = _drain_reader(t)
    assert [(f[0], f[1]) for f in frames] == [
        ("03_data_00001", 1), ("03_data_00001", 2),
        ("04_data_00002", 1), ("04_data_00002", 2), ("04_data_00002", 3),
    ]
    assert str(watch / "01_align_00001.nxs") in t._eiger_done_masters
    assert str(watch / "02_processed_00001.nxs") in t._eiger_done_masters


def test_single_file_nexus_wrangler_rejects_processed_cleanly(tmp_path):
    """Sibling caller of the shared finder: the single-file NeXus reduction
    wrangler (`open_nexus_image_stack`) must reject a processed xdart file with a
    clean, actionable stop — NOT let the finder's new ProcessedXdartInputError
    (a ValueError) escape run()'s try/finally as an uncaught QThread exception."""
    from xdart.gui.tabs.static_scan.wranglers.nexus_wrangler_thread import (
        nexusThread,
    )
    proc = _write_processed_xdart(tmp_path / "prev_run_00001.nxs")
    out = tmp_path / "out"
    out.mkdir()
    scan = LiveScan("scan", data_file=str(out / "scan.nxs"), static=True)
    t = nexusThread(
        Queue(), {}, threading.RLock(), str(out / "scan.nxs"),
        str(proc),                       # nexus_file — the processed output
        PONI(dist=0.2, poni1=0.1, poni2=0.1, wavelength=1e-10),
        None,                            # mask_file
        False, None, 1, 0.0,             # gi, th_mtr, sample_orientation, tilt
        "q_total", "qip_qoop",           # gi modes
        "start", scan,
    )
    labels = []
    t.showLabel.connect(labels.append)
    # Must return cleanly (before the diff: uncaught ProcessedXdartInputError).
    t._run_impl()
    assert any("processed xdart" in m.lower() for m in labels), labels
    # No output was written on rejection (rejection precedes any reduction/save).
    assert not (out / "scan.nxs").exists()


def test_processed_file_never_resolves_integrated_2d_via_worker(tmp_path):
    """Direct pin: opening a processed container through the worker's open path
    resolves NO dataset (nframes=0), never /entry/integrated_2d/intensity."""
    watch = tmp_path / "watch"
    out = tmp_path / "out"
    watch.mkdir()
    out.mkdir()
    proc = _write_processed_xdart(watch / "proc_00001.nxs")

    t = _make_thread(watch, out)
    t._eiger_open_master(str(proc))
    assert t._eiger_nframes == 0
    assert t._eiger_h5_dataset is None
    assert t._skip_reason_counts.get("processed xdart output", 0) == 1
