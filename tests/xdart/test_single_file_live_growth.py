# -*- coding: utf-8 -*-
"""Single-file live growth and nascent retry (NXS-SF-1 / NXS-SF-2).

Drives the REAL wrangler sync reader (``_get_next_eiger_frame_sync``) and the
real prefetch machinery over real HDF5 files — no fake cursors on the seam
under test.  Handoff §6/§7 (post_r2_nexus_source_edge_2026-07-19):

- a live single-file run that initially sees N frames and later grows to N+M
  publishes exactly N+M frames in order, continuing from the prior index
  (fail-before at 837397b9: frozen at N; regrown frames burned the tolerant
  read deadline through the fabio fallback and were skipped);
- a nascent NXWriter shell stays provisional (typed "not ready", never an
  "open error") and the same run processes frames that land later
  (fail-before: generic open error + the same skip treadmill);
- Stop interrupts the provisional wait promptly and leaves no open cursor;
- batch/non-live never waits on a shell (fixed point);
- a finalized consumed single file reaches a zero-open idle fixed point
  (fail-before: one full count_frames open per watch poll, forever).
"""

from __future__ import annotations

import os
import threading
import time
from collections import Counter, deque
from pathlib import Path

import h5py
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import (
    imageThread,
)


# ── fixtures ──────────────────────────────────────────────────────────────

def _nxwriter_shell(path):
    """A nascent Bluesky/NXWriter container: creator stamped, no detector
    tree, no end_time — the state a live watcher sees at run start."""
    with h5py.File(path, "w") as f:
        f.attrs["creator"] = "NXWriter"
        e = f.create_group("entry")
        e.attrs["NX_class"] = "NXentry"


def _nxwriter_frames(path, n, *, start=0):
    """Create-or-grow the detector stack to *n* frames (maxshape unlimited).

    Plain append opens suffice: the worker closes its cursor on every
    provisional 'wait' (it must never pin a file another program is writing),
    so the writer sees no other handle between polls."""
    with h5py.File(path, "a") as f:
        if "entry/instrument/detector/data" not in f:
            det = f["entry"].require_group("instrument").require_group("detector")
            det.create_dataset(
                "data", shape=(n, 8, 8), maxshape=(None, 8, 8),
                chunks=(1, 8, 8), dtype=np.uint16)
            ds = det["data"]
            lo = 0
        else:
            ds = f["entry/instrument/detector/data"]
            lo = ds.shape[0]
            ds.resize((n, 8, 8))
        for i in range(lo, n):
            ds[i] = np.full((8, 8), i + 1, dtype=np.uint16)


def _finalize(path):
    with h5py.File(path, "a") as f:
        f["entry"].create_dataset("end_time", data=b"2026-07-19T00:00:00")


def _plain_finalized(path, n):
    """A non-Bluesky container: finalized by definition (no NXWriter marker)."""
    with h5py.File(path, "w") as f:
        e = f.create_group("entry")
        e.attrs["NX_class"] = "NXentry"
        det = e.create_group("instrument").create_group("detector")
        det.create_dataset(
            "data",
            data=np.stack([np.full((8, 8), i + 1, dtype=np.uint16)
                           for i in range(n)]))


def _worker(path, *, live=True, batch=False):
    """A bare imageThread wired exactly like a single-file run (the same
    attribute set run() resets), driving the real production reader."""
    w = imageThread.__new__(imageThread)
    w.img_file = str(path)
    w.img_ext = "nxs"
    w.inp_type = "Image File"
    w.single_img = False
    w.live_mode = live
    w.batch_mode = batch
    w.command = ""
    w.write_mode = "Add"
    w.xye_only = False
    w.series_average = False
    w.meta_ext = None
    w.meta_dir = None
    w.FRAME_READ_DEADLINE = 1.0        # bound the tolerant-read wait for tests
    w._eiger_master_path = None
    w._eiger_frame_idx = 0
    w._eiger_nframes = 0
    w._eiger_master_queue = deque()
    w._eiger_done_masters = set()
    w._eiger_retry_after = {}
    w._eiger_zero_frame_seen = {}
    w._eiger_open_state = None
    w._eiger_cursor = None
    w._eiger_descriptor = None
    w._eiger_read_plan = None
    w._eiger_provider = None
    w._eiger_fabio_handle = None
    w._eiger_metadata_cache = {}
    w._bluesky_source_cache = {}
    w._prefetch_queue = None
    w._prefetch_thread = None
    w._prefetch_stop_evt = None
    w._prefetch_error = None
    w._perf = None
    w._discovered_frame_count = 0
    w._skip_reason_counts = Counter()
    w._append_skip_frames_by_scan = {}
    w._append_skip_without_reading = 0
    return w


def _drain(worker, limit=32):
    """Pull sync frames until the end-of-stream sentinel; return img numbers."""
    numbers = []
    for _ in range(limit):
        _path, _scan, number, data, _meta = worker._get_next_eiger_frame_sync()
        if data is None:
            return numbers
        numbers.append(number)
    raise AssertionError(f"no end-of-stream sentinel after {limit} frames")


def _count_cursor_opens(monkeypatch):
    """Observe (not fake) ContainerCursor.open calls on the production class."""
    from xrd_tools.sources.cursor import ContainerCursor

    calls = []
    real_open = ContainerCursor.open

    def counting_open(self):
        calls.append(str(getattr(self, "_path", "")))
        return real_open(self)

    monkeypatch.setattr(ContainerCursor, "open", counting_open)
    return calls


# ── NXS-SF-2: nascent shell stays provisional, then processes ─────────────

def test_single_file_nascent_shell_then_frames(tmp_path):
    path = tmp_path / "scan_00001.nxs"
    _nxwriter_shell(path)
    worker = _worker(path)

    assert _drain(worker) == []          # provisional: nothing yet, no crash
    assert worker._eiger_open_state == "not ready", \
        "a nascent shell is typed provisional, not an open error (NXS-SF-2)"
    assert worker._eiger_master_path is not None, \
        "the provisional master must stay armed for the same run"

    _nxwriter_frames(path, 3)            # detector tree lands mid-run

    got = []
    deadline = time.monotonic() + 15.0
    while len(got) < 3 and time.monotonic() < deadline:
        got.extend(_drain(worker))       # each drain = one watch poll
    assert got == [1, 2, 3], \
        f"the SAME run must process frames that land later, got {got}"
    assert worker._skip_reason_counts.get(
        "unreadable or empty image data", 0) == 0, \
        "landed frames must be read, not burned on the tolerant-read deadline"


# ── NXS-SF-1: growth publishes exactly N+M in order ──────────────────────

def test_single_file_growth_exactly_once_in_order(tmp_path, monkeypatch):
    opens = _count_cursor_opens(monkeypatch)
    path = tmp_path / "grow_00001.nxs"
    _nxwriter_shell(path)
    _nxwriter_frames(path, 3)
    worker = _worker(path)

    got = _drain(worker)                 # initial segment
    assert got == [1, 2, 3]

    _nxwriter_frames(path, 5)            # +2 frames appended (non-SWMR)

    deadline = time.monotonic() + 15.0
    while len(got) < 5 and time.monotonic() < deadline:
        got.extend(_drain(worker))
    assert got == [1, 2, 3, 4, 5], \
        f"growth must continue from the prior index, exactly once: {got}"

    _finalize(path)                      # end_time lands -> fixed point
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        assert _drain(worker) == []
        if worker._eiger_cursor is None:
            break
    assert worker._eiger_cursor is None, \
        "a finalized consumed file must close its cursor (fixed point)"
    assert len(opens) <= 12, \
        f"growth polling must stay bounded; observed {len(opens)} cursor opens"


# ── Stop while provisional ────────────────────────────────────────────────

def test_single_file_stop_while_provisional_joins_clean(tmp_path, monkeypatch):
    path = tmp_path / "stop_00001.nxs"
    _nxwriter_shell(path)
    worker = _worker(path)

    t0 = time.monotonic()
    _p, _s, _n, data, _m = worker._get_next_eiger_frame()   # real prefetcher
    assert data is None
    # The provisional path itself must leave NO open cursor behind — asserted
    # BEFORE any test-side cleanup (the close-on-wait production invariant).
    assert worker._eiger_cursor is None, \
        "a provisional wait must not hold an open cursor between polls"
    worker.command = "stop"
    stop_evt = worker._prefetch_stop_evt
    if stop_evt is not None:
        stop_evt.set()
    assert worker._prefetch_stop_prior(), "prefetch worker must join promptly"
    thread = worker._prefetch_thread
    assert thread is None or not thread.is_alive()
    assert time.monotonic() - t0 < 10.0, "Stop must not wait out a deadline"

    # A stopped worker's polls must not open the source at all.
    opens = _count_cursor_opens(monkeypatch)
    assert _drain(worker) == []
    assert len(opens) == 0, "a stopped run must not reopen the source"
    worker._eiger_close_master()                 # end-of-run hygiene (run())


# ── Non-live fixed point ──────────────────────────────────────────────────

def test_single_file_batch_never_waits_on_shell(tmp_path):
    path = tmp_path / "batch_00001.nxs"
    _nxwriter_shell(path)
    worker = _worker(path, live=False, batch=True)

    t0 = time.monotonic()
    assert _drain(worker) == []
    assert time.monotonic() - t0 < 5.0, \
        "batch must not wait for a shell to grow (NXS-SF-2 non-live)"


def test_single_file_batch_unfinalized_reads_then_ends(tmp_path):
    path = tmp_path / "batchgrow_00001.nxs"
    _nxwriter_shell(path)
    _nxwriter_frames(path, 3)
    worker = _worker(path, live=False, batch=True)

    t0 = time.monotonic()
    assert _drain(worker) == [1, 2, 3]
    assert time.monotonic() - t0 < 10.0, \
        "batch consumes what exists and ends without a growth wait"


def _append_frames_subprocess(path, n):
    """Append to *n* frames from a SEPARATE process — the beamline writer's
    actual shape.  In-process HDF5 refuses a mixed RDONLY/RDWR open of the
    same file, and the point here is growth landing while the worker's read
    cursor is STILL OPEN; ``locking=False`` bypasses the reader's advisory
    lock exactly as an external writer configuration does."""
    import subprocess
    import sys

    script = (
        "import h5py, numpy as np\n"
        f"with h5py.File({str(path)!r}, 'a', locking=False) as f:\n"
        "    ds = f['entry/instrument/detector/data']\n"
        "    lo = ds.shape[0]\n"
        f"    ds.resize(({n}, 8, 8))\n"
        f"    for i in range(lo, {n}):\n"
        "        ds[i] = np.full((8, 8), i + 1, dtype=np.uint16)\n"
    )
    subprocess.run([sys.executable, "-c", script],
                   check=True, capture_output=True)


def test_single_file_growth_while_cursor_open_uses_grown_arm(tmp_path):
    """Growth landing DURING consumption (the common live cadence): at
    exhaustion the still-open cursor's cached count is stale, and the
    transactional-reopen 'grown' arm — not the cursor-less count recheck —
    must observe the tail and continue from the prior index."""
    path = tmp_path / "growopen_00001.nxs"
    _nxwriter_shell(path)
    _nxwriter_frames(path, 3)
    worker = _worker(path)

    outcomes = []
    real_outcome = worker._eiger_single_file_growth_outcome

    def spying_outcome():
        out = real_outcome()
        outcomes.append(out)
        return out

    worker._eiger_single_file_growth_outcome = spying_outcome

    got = []
    for _ in range(3):                           # consume the initial segment
        _p, _s, num, data, _m = worker._get_next_eiger_frame_sync()
        assert data is not None
        got.append(num)
    assert got == [1, 2, 3]
    assert worker._eiger_cursor is not None, "consumption keeps ONE cursor"

    _append_frames_subprocess(path, 5)           # tail lands while it is open

    for _ in range(2):
        _p, _s, num, data, _m = worker._get_next_eiger_frame_sync()
        assert data is not None, "the grown tail must be read, not sentineled"
        got.append(num)
    assert got == [1, 2, 3, 4, 5]
    assert "grown" in outcomes, \
        f"the transactional-reopen growth arm never executed: {outcomes}"


# ── fabio-primary live master: catch-up must not latch the run done ───────

def test_fabio_master_catchup_stays_provisional(tmp_path):
    """fabio's nframes counts only LANDED data files, so a live reader that
    catches up with the writer sees 'exhaustion' while a declared segment is
    still in flight.  That must classify 'wait' (retry on the next watch
    poll) — latching done would permanently end source reads mid-acquisition
    (review-caught blocker)."""
    master = tmp_path / "fab_master.h5"
    t1 = tmp_path / "fab_data_000001.h5"
    t2 = tmp_path / "fab_data_000002.h5"
    with h5py.File(t1, "w") as f:
        f.create_dataset(
            "entry/data/data",
            data=np.arange(3 * 64, dtype=np.uint16).reshape(3, 8, 8))
    with h5py.File(master, "w") as f:            # segment 2 NOT landed yet
        e = f.create_group("entry")
        e.attrs["NX_class"] = "NXentry"
        d = e.create_group("data")
        d.attrs["NX_class"] = "NXdata"
        d["data_000001"] = h5py.ExternalLink(str(t1), "/entry/data/data")
        d["data_000002"] = h5py.ExternalLink(str(t2), "/entry/data/data")

    worker = _worker(master)
    worker.img_ext = "h5"

    _p, _s, num, data, _m = worker._get_next_eiger_frame_sync()
    assert data is not None and num == 1
    assert worker._eiger_fabio_handle is not None, \
        "precondition: fabio must own this master (else the test is moot)"

    got = [num] + _drain(worker)                 # catch up with the writer
    assert got == [1, 2, 3]
    assert not getattr(worker, "_eiger_single_file_done", False), \
        "a live fabio catch-up must stay provisional, never latch done"
    assert worker._eiger_single_file_watchable(), \
        "the watch gate must keep polling a caught-up live fabio master"

    with h5py.File(t2, "w") as f:                # segment 2 lands
        f.create_dataset(
            "entry/data/data",
            data=np.arange(3 * 64, 6 * 64, dtype=np.uint16).reshape(3, 8, 8))

    deadline = time.monotonic() + 15.0
    while len(got) < 6 and time.monotonic() < deadline:
        got.extend(_drain(worker))
    assert got == [1, 2, 3, 4, 5, 6], \
        f"the run must resume when the declared segment lands, got {got}"


# ── Phase-3 watch gate facts survive the provisional close ────────────────

def test_watch_gate_facts_from_production_reader_states(tmp_path):
    """_eiger_single_file_watchable consumes state exactly as the production
    reader leaves it (the provisional close clears the descriptor — the gate
    must not depend on it)."""
    # nascent shell -> watchable
    shell = tmp_path / "gate_shell_00001.nxs"
    _nxwriter_shell(shell)
    w1 = _worker(shell)
    assert _drain(w1) == []
    assert w1._eiger_single_file_watchable(), \
        "a nascent shell must keep the live watch alive (NXS-SF-2)"

    # created-but-EMPTY detector dataset, unfinalized -> watchable
    # (open succeeds, so open_state is 'ready' — the provisional FLAG, not
    # the cleared descriptor, must carry the fact; review-caught)
    empty = tmp_path / "gate_empty_00001.nxs"
    _nxwriter_shell(empty)
    with h5py.File(empty, "a") as f:
        det = f["entry"].require_group("instrument").require_group("detector")
        det.create_dataset("data", shape=(0, 8, 8), maxshape=(None, 8, 8),
                           chunks=(1, 8, 8), dtype=np.uint16)
    w2 = _worker(empty)
    assert _drain(w2) == []
    assert w2._eiger_single_file_watchable(), \
        "an unfinalized zero-frame detector dataset is provisional, not done"

    # plain finalized, fully consumed -> NOT watchable (fixed point)
    done = tmp_path / "gate_done_00001.nxs"
    _plain_finalized(done, 2)
    w3 = _worker(done)
    assert _drain(w3) == [1, 2]
    assert _drain(w3) == []
    assert not w3._eiger_single_file_watchable(), \
        "a finalized consumed file must not watch (fixed point)"


# ── Transient sharing denial stays provisional (§10, Windows-style) ───────

def test_single_file_transient_denial_stays_provisional(tmp_path, monkeypatch):
    """A PermissionError during the growth reopen (a writer holding the file,
    the Windows sharing-denial shape — no platform assumption, the exception
    type is the contract) must classify 'wait', never latch the run done."""
    from xrd_tools.sources.cursor import ContainerCursor

    path = tmp_path / "deny_00001.nxs"
    _nxwriter_shell(path)
    _nxwriter_frames(path, 3)
    worker = _worker(path)
    assert _drain(worker) == [1, 2, 3]

    real_open = ContainerCursor.open
    denials = {"n": 0}

    def denying_open(self):
        if denials["n"] == 0:
            denials["n"] += 1
            raise PermissionError(13, "sharing violation", str(path))
        return real_open(self)

    monkeypatch.setattr(ContainerCursor, "open", denying_open)
    assert _drain(worker) == []          # denied poll: provisional, no crash
    assert not getattr(worker, "_eiger_single_file_done", False), \
        "a transient denial must not latch the single-file run done"

    _nxwriter_frames(path, 4)            # writer releases; growth lands
    got = []
    deadline = time.monotonic() + 15.0
    while len(got) < 1 and time.monotonic() < deadline:
        got.extend(_drain(worker))
    assert got == [4], f"the run must recover after the denial, got {got}"


# ── Finalized single file: zero-open idle fixed point ─────────────────────

def test_single_file_finalized_idle_is_zero_open(tmp_path, monkeypatch):
    from xdart.gui.tabs.static_scan.wranglers import image_wrangler_thread

    path = tmp_path / "done_00001.nxs"
    _plain_finalized(path, 3)
    worker = _worker(path)

    assert _drain(worker) == [1, 2, 3]

    opens = _count_cursor_opens(monkeypatch)
    counts = []
    real_count = image_wrangler_thread.count_frames

    def counting_count(p, *a, **k):
        counts.append(str(p))
        return real_count(p, *a, **k)

    monkeypatch.setattr(image_wrangler_thread, "count_frames", counting_count)

    for _ in range(5):                   # five idle watch polls
        assert _drain(worker) == []
    assert len(opens) == 0 and len(counts) == 0, \
        (f"a finalized consumed single file must idle with ZERO source opens; "
         f"observed {len(opens)} cursor opens, {len(counts)} count_frames")
