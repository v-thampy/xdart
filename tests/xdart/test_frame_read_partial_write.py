"""Live/network-share partial-write tolerance in the image wrangler.

Beamline regression: the detector CREATES a TIF on the SMB/NFS share and is
still flushing it when xdart's glob picks it up, so fabio hits an empty/truncated
file and raises ``OSError: Fabio could not identify`` (or an ``IndexError`` a few
bytes later).  Unhandled, that escaped ``get_next_image`` -> ``process_scan`` ->
``run()`` and KILLED the live QThread -- live processing stopped dead,
indistinguishable from a timeout to the user.

Two layers of tolerance, both exercised here against REAL fabio on real files:
  * ``_read_frame_tolerant`` -- never lets the read raise; short retry absorbs a
    file that finishes flushing within ~a second; returns None otherwise.
  * ``get_next_image`` -- on None in LIVE mode it re-polls the frame on later
    sweeps WITHOUT committing/dropping it (so a slow write is read, not lost),
    until a wall-clock deadline; only then (or in batch) does it skip.

No fakes on the read seam being fixed.  Imports are deferred into the tests (they
pull the static_scan GUI stack; keep collection headless-safe, per test_gui_logging).
"""
import types
from collections import deque

import fabio
import fabio.tifimage        # explicit: fabio does not eagerly load submodules
import numpy as np


def _imageThread():
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import imageThread
    return imageThread


def _write_valid_tif(path, shape=(16, 16)):
    """Write a real TIF via fabio itself, so fabio is guaranteed to read it
    back -- returns the full byte payload for truncation games."""
    data = np.arange(int(np.prod(shape)), dtype=np.int32).reshape(shape)
    fabio.tifimage.TifImage(data=data).write(str(path))
    return path.read_bytes()


# --------------------------------------------------------------------------- #
# _read_frame_tolerant (the safe read: never raises)                          #
# --------------------------------------------------------------------------- #
def _read_stub(budget=0.1, command="live"):
    return types.SimpleNamespace(command=command, FRAME_READ_RETRY_BUDGET=budget)


def _tolerant(stub, path):
    return _imageThread()._read_frame_tolerant(stub, str(path))


def test_tolerant_read_reads_a_valid_tif(tmp_path):
    p = tmp_path / "frame_0000.tif"
    _write_valid_tif(p)
    arr = _tolerant(_read_stub(), p)
    assert arr is not None and arr.shape == (16, 16)


def test_tolerant_read_returns_none_not_raise_on_empty_file(tmp_path):
    p = tmp_path / "frame_0001.tif"
    _write_valid_tif(p)
    p.write_bytes(b"")                               # created, not flushed -> fabio raises
    assert _tolerant(_read_stub(budget=0.2), p) is None   # returns, does NOT raise


def test_tolerant_read_returns_immediately_on_stop(tmp_path):
    p = tmp_path / "frame_0002.tif"
    _write_valid_tif(p)
    p.write_bytes(b"")
    assert _tolerant(_read_stub(budget=60.0, command="stop"), p) is None


# --------------------------------------------------------------------------- #
# get_next_image (cross-sweep re-poll: never drop a slow write)               #
# --------------------------------------------------------------------------- #
def _series_stub(fnames, *, batch_mode, budget=0.05, deadline=30.0):
    it = _imageThread()
    stub = types.SimpleNamespace(
        single_img=False, img_file=None, img_ext="tif",
        img_fnames=deque(str(f) for f in fnames),
        processed=[], _frame_read_clocks={},
        batch_mode=batch_mode, series_average=False, meta_ext="",
        command="live",
        FRAME_READ_RETRY_BUDGET=budget, FRAME_READ_DEADLINE=deadline,
    )
    stub._read_frame_tolerant = it._read_frame_tolerant.__get__(stub)
    stub._commit_frame = it._commit_frame.__get__(stub)
    stub._frame_read_clock_map = it._frame_read_clock_map.__get__(stub)
    stub._frame_read_deadline_reached = it._frame_read_deadline_reached.__get__(stub)
    stub._should_skip_before_read = lambda *a, **k: False
    stub._record_discovered_frame = lambda *a, **k: None
    stub._record_skip_reason = lambda *a, **k: None
    return stub


def test_live_repolls_partial_frame_and_does_not_drop_it(tmp_path):
    # A frame created-but-not-flushed must be RE-POLLED (not committed/dropped)
    # and then READ once the write completes -- the core anti-frame-loss claim.
    p = tmp_path / "frame_0000.tif"
    good = _write_valid_tif(p)
    p.write_bytes(b"")                               # detector created it, no data yet
    stub = _series_stub([p], batch_mode=False, deadline=30.0)

    r1 = _imageThread().get_next_image(stub)         # sweep 1: not ready
    assert r1[3] is None                             # img_data None -> watch re-polls
    assert str(p) not in stub.processed              # NOT committed / dropped
    assert len(stub.img_fnames) == 1                 # still queued for retry

    p.write_bytes(good)                              # the write completes
    r2 = _imageThread().get_next_image(stub)         # sweep 2: now reads it
    assert r2[3] is not None and r2[3].shape == (16, 16)
    assert str(p) in stub.processed                  # now committed exactly once


def test_live_skips_frame_once_past_deadline(tmp_path):
    # A genuinely corrupt file must not wedge live forever: once the deadline
    # passes it is skipped and the next good frame is read.
    corrupt = tmp_path / "frame_0000.tif"
    _write_valid_tif(corrupt); corrupt.write_bytes(b"")
    goodf = tmp_path / "frame_0001.tif"
    _write_valid_tif(goodf)
    stub = _series_stub([corrupt, goodf], batch_mode=False, deadline=0.0)

    r1 = _imageThread().get_next_image(stub)         # starts the clock, re-polls
    assert r1[3] is None
    r2 = _imageThread().get_next_image(stub)         # deadline reached -> skip corrupt, read good
    assert r2[3] is not None
    assert str(corrupt) in stub.processed


def test_batch_skips_unreadable_frame_without_repoll(tmp_path):
    # Batch files are already complete, so an unreadable one is skipped in a
    # single pass (no re-poll) and the batch continues -- never aborts.
    corrupt = tmp_path / "frame_0000.tif"
    _write_valid_tif(corrupt); corrupt.write_bytes(b"")
    goodf = tmp_path / "frame_0001.tif"
    _write_valid_tif(goodf)
    stub = _series_stub([corrupt, goodf], batch_mode=True)

    r = _imageThread().get_next_image(stub)          # corrupt skipped, good returned, no raise
    assert r[3] is not None
    assert str(corrupt) in stub.processed


# --------------------------------------------------------------------------- #
# Eiger: read must wait for a data file that lags the master (not exit)       #
# --------------------------------------------------------------------------- #
# An Eiger master declares nimages at scan start, but the per-frame data files
# stream in as the detector writes them.  Reading a declared frame whose data
# has not landed used to raise -> the worker read that as end-of-stream and
# exited -> live Eiger stalled.  _read_eiger_frame_tolerant must wait (refreshing
# the handle) until the data lands, then read it -- exercised on the REAL h5py
# read path with a resizable dataset that grows to simulate the data landing.
def _eiger_stub(dset, master_path, *, batch_mode, deadline=30.0, command="live"):
    import threading
    stub = types.SimpleNamespace(
        command=command, batch_mode=batch_mode,
        _prefetch_stop_evt=threading.Event(),
        _eiger_fabio_handle=None, _eiger_h5_dataset=dset,
        _eiger_master_path=str(master_path), _eiger_nframes=dset.shape[0],
        FRAME_READ_DEADLINE=deadline,
    )
    stub._eiger_refresh_master_handle = _imageThread()._eiger_refresh_master_handle.__get__(stub)
    return stub


def test_eiger_read_waits_for_lagging_data_frame_then_reads(tmp_path, monkeypatch):
    import h5py
    it = _imageThread()
    f = tmp_path / "master.h5"
    with h5py.File(f, "w") as h:
        h.create_dataset("d", data=np.zeros((3, 8, 8), "i4"), maxshape=(None, 8, 8))
    h = h5py.File(f, "r+")
    try:
        dset = h["d"]
        stub = _eiger_stub(dset, f, batch_mode=False, deadline=30.0)

        import xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread as mod

        def fake_sleep(_dt):
            if dset.shape[0] == 3:
                dset.resize((10, 8, 8))          # the lagging data file lands
        monkeypatch.setattr(mod.time, "sleep", fake_sleep)

        arr = it._read_eiger_frame_tolerant(stub, 5)   # frame 5 absent -> wait -> read
        assert arr is not None and arr.shape == (8, 8)
    finally:
        h.close()


def test_eiger_read_batch_returns_none_without_waiting(tmp_path):
    import h5py
    it = _imageThread()
    f = tmp_path / "master.h5"
    with h5py.File(f, "w") as h:
        h.create_dataset("d", data=np.zeros((3, 8, 8), "i4"))
    h = h5py.File(f, "r")
    try:
        stub = _eiger_stub(h["d"], f, batch_mode=True)
        assert it._read_eiger_frame_tolerant(stub, 5) is None   # missing frame, no retry, no raise
    finally:
        h.close()


def test_eiger_read_returns_none_on_stop(tmp_path):
    import h5py
    it = _imageThread()
    f = tmp_path / "master.h5"
    with h5py.File(f, "w") as h:
        h.create_dataset("d", data=np.zeros((3, 8, 8), "i4"))
    h = h5py.File(f, "r")
    try:
        stub = _eiger_stub(h["d"], f, batch_mode=False, command="stop", deadline=60.0)
        assert it._read_eiger_frame_tolerant(stub, 5) is None
    finally:
        h.close()
