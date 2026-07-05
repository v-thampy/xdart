# -*- coding: utf-8 -*-
"""Regression tests for the persist-before-evict data-loss fix.

Bug: ``LiveFrameSeries`` kept integrated frames in a 64-entry FIFO cache and
evicted the oldest with NO persist.  The v2 writer reads ``int_1d``/``int_2d``
straight off these in-memory objects, so when ``LIVE_SAVE_INTERVAL`` exceeded
the cap on a scan longer than the cap, frames were evicted before any save and
their integration results were silently lost (empty/absent ``integrated_1d``).

Fix: ``stash`` only evicts frames marked ``mark_persisted`` (written to disk);
unsaved frames are never dropped.  The non-batch dispatcher additionally forces
a save before the unsaved set reaches the cap (see ``_save_due``).
"""

from __future__ import annotations

import threading
import logging

import numpy as np
import pytest


def _result_1d(idx, nq=16):
    from xrd_tools.core.containers import IntegrationResult1D
    return IntegrationResult1D(
        radial=np.linspace(0.5, 5.0, nq, dtype=np.float32),
        intensity=np.full(nq, float(idx + 1), dtype=np.float32),
        sigma=np.ones(nq, dtype=np.float32),
        unit="q_A^-1",
    )


def _make_frame(idx):
    from xdart.modules.ewald.frame import LiveFrame
    fr = LiveFrame(idx=idx)
    fr.int_1d = _result_1d(idx)
    fr.scan_info = {"i0": float(idx + 1)}
    fr.source_file = ""        # no on-disk source; int_1d lives only on disk
    fr.source_frame_idx = 0
    fr.skip_map_raw = True
    return fr


# ---------------------------------------------------------------------------
# Unit: LiveFrameSeries eviction policy
# ---------------------------------------------------------------------------

class TestStashPersistBeforeEvict:
    def _series(self, cap=4):
        from xdart.modules.ewald.frame_series import LiveFrameSeries
        fs = LiveFrameSeries(data_file="", file_lock=threading.RLock())
        fs._in_memory_cap = cap
        return fs

    def test_unsaved_frames_never_evicted_even_past_cap(self):
        from xdart.modules.ewald.frame import LiveFrame
        fs = self._series(cap=4)
        for i in range(10):
            fs[i] = LiveFrame(idx=i)
        # Nothing persisted -> nothing may be evicted, so the cache exceeds the
        # cap rather than drop an unsaved frame.
        assert len(fs._in_memory) == 10
        assert fs.unsaved_in_memory_count() == 10
        for i in range(10):
            assert i in fs._in_memory

    def test_only_persisted_frames_are_evicted(self):
        from xdart.modules.ewald.frame import LiveFrame
        fs = self._series(cap=4)
        for i in range(6):
            fs[i] = LiveFrame(idx=i)
        fs.mark_persisted(range(6))          # 0..5 now on disk
        for i in range(6, 12):
            fs[i] = LiveFrame(idx=i)          # 6..11 unsaved
        # Every unsaved frame survives.
        for i in range(6, 12):
            assert i in fs._in_memory
        assert fs.unsaved_in_memory_count() == 6
        # Persisted frames were evicted to keep the cache bounded.
        assert all(i not in fs._in_memory for i in range(6)) or \
            len(fs._in_memory) <= 6 + fs._in_memory_cap

    def test_unmark_persisted_reverts_evictability(self):
        """Cluster B: a dropped reintegrate shadow's frames must be un-marked so
        they are no longer treated as safely-on-canonical (they only lived in
        the deleted shadow).  unmark only makes frames LESS evictable."""
        from xdart.modules.ewald.frame import LiveFrame
        fs = self._series(cap=4)
        for i in range(8):
            fs[i] = LiveFrame(idx=i)
        fs.mark_persisted(range(8))
        # The reintegrate "Stop" path discards the shadow for a subset.
        fs.unmark_persisted([2, 3, 4])
        assert {2, 3, 4}.isdisjoint(fs._persisted)
        assert {0, 1, 5, 6, 7} <= fs._persisted
        # A still-resident unmarked frame is no longer eviction-eligible.
        fs._in_memory = {2: LiveFrame(idx=2)}
        assert fs.evict_persisted_beyond_cap() == 0

    def test_discard_in_memory_forces_lazy_reload(self):
        """Cluster B: discarding a stopped reintegrate's in-memory recomputed
        frame makes the next access lazy-load (prior canonical) instead of
        returning the abandoned recomputed object."""
        from xdart.modules.ewald.frame import LiveFrame
        fs = self._series(cap=4)
        for i in range(3):
            fs[i] = LiveFrame(idx=i)
        assert 1 in fs._in_memory
        fs.discard_in_memory([1])
        assert 1 not in fs._in_memory       # gone -> next __getitem__ re-loads
        assert 0 in fs._in_memory and 2 in fs._in_memory

    def test_evict_persisted_beyond_cap_releases_after_save(self):
        """D1: after a reintegrate's single end-of-run save, no further stash
        fires to trigger eviction, so the just-saved frames would stay pinned.
        evict_persisted_beyond_cap releases the persisted ones down to the cap --
        but never an unsaved frame."""
        from xdart.modules.ewald.frame import LiveFrame
        fs = self._series(cap=4)
        for i in range(20):
            fs[i] = LiveFrame(idx=i)
        assert len(fs._in_memory) == 20          # nothing persisted yet -> pinned
        fs.mark_persisted(range(20))             # the reintegrate save persisted all
        n = fs.evict_persisted_beyond_cap()
        assert n == 20 - fs._in_memory_cap       # 16 released
        assert len(fs._in_memory) == fs._in_memory_cap
        # An unsaved frame is never dropped by the sweep, even past cap.
        fs[100] = LiveFrame(idx=100)             # unsaved
        fs.evict_persisted_beyond_cap()
        assert 100 in fs._in_memory
        assert fs.unsaved_in_memory_count() == 1

    def test_stash_marks_frame_unsaved_again(self):
        from xdart.modules.ewald.frame import LiveFrame
        fs = self._series()
        fs[0] = LiveFrame(idx=0)
        fs.mark_persisted([0])
        assert fs.unsaved_in_memory_count() == 0
        # Re-stash (e.g. reintegration produces new int_1d) -> dirty again.
        fs[0] = LiveFrame(idx=0)
        assert fs.unsaved_in_memory_count() == 1


# ---------------------------------------------------------------------------
# Integration: N > cap, single end-of-run save, reload has every frame
# ---------------------------------------------------------------------------

class TestNonBatchNoDataLoss:
    def test_reload_has_all_frames_when_interval_exceeds_cap(self, tmp_path):
        """The reported bug: cap < N < interval (one save at the end).  Without
        persist-before-evict the early frames are FIFO-evicted before the save
        and lost; with it, all N survive in memory until the save writes them."""
        from xdart.modules.ewald import LiveScan

        nxs = str(tmp_path / "scan.nxs")
        scan = LiveScan(data_file=nxs)
        scan.skip_2d = True
        scan.frames._in_memory_cap = 8       # small cap to force the scenario
        N = 25                                # 8 (cap) < 25 (N) < 1000 (interval)

        for i in range(N):
            scan.add_frame(frame=_make_frame(i), calculate=False,
                           update=True, get_sd=True, batch_save=True)
        # Only one save, at the very end — exactly what interval=1000 produces
        # for a 25-frame scan.
        scan._save_to_nexus()

        reloaded = LiveScan(data_file=nxs)
        reloaded.load_from_h5()
        assert len(reloaded.frames.index) == N
        for i in range(N):
            fr = reloaded.frames[i]
            assert fr.int_1d is not None, f"frame {i} lost its int_1d"
            np.testing.assert_allclose(
                np.asarray(fr.int_1d.intensity)[0], float(i + 1), rtol=0, atol=1e-4,
            )

    def test_periodic_saves_allow_eviction_and_lazy_reload(self, tmp_path):
        """With periodic saves, persisted frames evict + lazy-reload from disk,
        so the cache stays bounded AND no data is lost."""
        from xdart.modules.ewald import LiveScan

        nxs = str(tmp_path / "scan.nxs")
        scan = LiveScan(data_file=nxs)
        scan.skip_2d = True
        scan.frames._in_memory_cap = 8
        N = 25
        for i in range(N):
            scan.add_frame(frame=_make_frame(i), calculate=False,
                           update=True, get_sd=True, batch_save=True)
            if (i + 1) % 5 == 0:             # save every 5 frames
                scan._save_to_nexus()
        scan._save_to_nexus()
        # Cache bounded near the cap (persisted frames evicted).
        assert len(scan.frames._in_memory) <= scan.frames._in_memory_cap + 5

        reloaded = LiveScan(data_file=nxs)
        reloaded.load_from_h5()
        assert len(reloaded.frames.index) == N
        for i in range(N):
            assert reloaded.frames[i].int_1d is not None


class TestLoadFrameIndexOnly:
    """load_frame_index_only is the post-live reintegrate hookup: rebuild the lazy
    frame index from the finished .nxs WITHOUT a full set_datafile reload."""

    def test_populates_frames_without_clobbering_scan_state(self, tmp_path):
        import pandas as pd
        from xdart.modules.ewald import LiveScan

        nxs = str(tmp_path / "scan.nxs")
        writer = LiveScan(data_file=nxs)
        writer.skip_2d = True
        N = 6
        for i in range(N):
            writer.add_frame(frame=_make_frame(i), calculate=False,
                             update=True, get_sd=True, batch_save=True)
        writer._save_to_nexus()

        # A fresh scan standing in for the post-live state: frames empty (the
        # streaming path never populated them), but other scan state + the GUI
        # caches are live.  Sentinels here must survive the index load.
        scan = LiveScan(data_file=None)
        scan.bai_1d_args = {"numpoints": 999}
        scan.bai_2d_args = {"npt_rad": 777}
        scan.global_mask = np.array([1, 2, 3])
        scan.scan_data = pd.DataFrame({"i0": [10.0, 20.0]})
        assert not scan.frames.index            # empty before the load

        n = scan.load_frame_index_only(nxs)

        assert n == N
        assert list(scan.frames.index) == list(range(N))
        # Frames are usable + lazy: a frame loads its int_1d from disk on demand.
        assert scan.frames[0].int_1d is not None
        # And nothing else was touched (no reset / reload of these fields).
        assert scan.bai_1d_args == {"numpoints": 999}
        assert scan.bai_2d_args == {"npt_rad": 777}
        np.testing.assert_array_equal(scan.global_mask, [1, 2, 3])
        assert list(scan.scan_data["i0"]) == [10.0, 20.0]

    def test_missing_file_is_a_safe_noop(self, tmp_path):
        from xdart.modules.ewald import LiveScan
        scan = LiveScan(data_file=None)
        assert scan.load_frame_index_only(str(tmp_path / "nope.nxs")) == 0
        assert not scan.frames.index


def test_missing_integrated_row_logs_error_when_file_idle(tmp_path, caplog):
    import h5py
    from xdart.modules.ewald.frame_series import LiveFrameSeries

    nxs = tmp_path / "partial.nxs"
    with h5py.File(nxs, "w") as f:
        g = f.create_group("entry/integrated_1d")
        g.create_dataset("frame_index", data=np.array([1], dtype=np.int64))

    fs = LiveFrameSeries(str(nxs), threading.RLock())
    fs.index.append(1)

    with caplog.at_level(logging.ERROR, logger="xdart.modules.ewald.frame_series"):
        frame = fs[1]

    assert frame.idx == 1
    assert "indexed but has no integrated data on disk" in caplog.text


def test_missing_integrated_row_is_debug_while_rewrite_active(tmp_path, caplog):
    import h5py
    from xdart.modules.ewald.frame_series import LiveFrameSeries

    nxs = tmp_path / "rewriting.nxs"
    with h5py.File(nxs, "w") as f:
        g = f.create_group("entry/integrated_1d")
        g.create_dataset("frame_index", data=np.array([1], dtype=np.int64))

    fs = LiveFrameSeries(str(nxs), threading.RLock())
    fs.index.append(1)
    fs.set_integrated_reads_transient(True)

    with caplog.at_level(logging.DEBUG, logger="xdart.modules.ewald.frame_series"):
        frame = fs[1]

    assert frame.idx == 1
    assert "no integrated data on disk yet" in caplog.text
    assert not [record for record in caplog.records if record.levelno >= logging.ERROR]


def test_reload_probe_resolves_relative_source_against_source_base(tmp_path):
    """N1 (live regression 2026-06-18): source_file is stored RELATIVE to the
    project root (entry/@source_base), which is NOT the .nxs dir — the default
    output dir is <root>/xdart_processed_data.  has_reload_only_frames's probe
    must resolve against @source_base (like the real lazy load in _load_frame_v2),
    else it false-reports 'raw gone' and blocks a reintegrable scan."""
    import os
    from types import SimpleNamespace
    import h5py
    from xdart.modules.ewald import LiveScan

    root = tmp_path / "project"
    (root / "raw").mkdir(parents=True)
    (root / "raw" / "img.tif").write_bytes(b"x")        # the raw IS present
    proc = root / "xdart_processed_data"
    proc.mkdir()
    nxs = proc / "scan.nxs"
    with h5py.File(nxs, "w") as f:
        e = f.create_group("entry")
        e.attrs["source_base"] = str(root)              # N1 base = project root
        src = e.create_group("frames/frame_0000/source")
        src.create_dataset("path", data="raw/img.tif")  # RELATIVE to @source_base

    scan = LiveScan(data_file=str(nxs))
    scan.frames = SimpleNamespace(index=[0])            # force the Path-B h5 probe

    # Resolves "raw/img.tif" against @source_base (root) → exists → reachable.
    assert scan.has_reload_only_frames() is False
    # The same relpath under the .nxs dir does NOT exist — so resolving against
    # dirname(.nxs) (the old probe) would have wrongly reported the raw as gone.
    assert not os.path.exists(os.path.join(str(proc), "raw", "img.tif"))
