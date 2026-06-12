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
