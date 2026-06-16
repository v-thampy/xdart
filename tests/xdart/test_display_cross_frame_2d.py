# -*- coding: utf-8 -*-
"""Cross-frame 2D readers: on-demand disk hydration + the axis-identity guard.

Bug (shipped): ``data_2d`` is a bounded window (FixSizeOrderedDict max=20), but
``get_frames_int_2d``/``get_frames_map_raw`` iterate the selection and skip a
``data_2d`` miss — so a 2D sum/average / Set-Bkg over a selection larger than
the cache silently averaged only the cached subset.  Fix: hydrate evicted frames
from disk.  Plus an axis-identity guard so the reducer never sums misaligned
cakes (the writer enforces within-scan uniform axes; the guard makes it explicit).
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest


def _ir1d(val, nq=8):
    from xrd_tools.core.containers import IntegrationResult1D
    return IntegrationResult1D(
        radial=np.linspace(0.5, 5.0, nq, dtype=np.float32),
        intensity=np.full(nq, float(val), dtype=np.float32),
        sigma=np.ones(nq, dtype=np.float32), unit="q_A^-1")


def _ir2d(val, *, nq=8, nchi=6, rad=None):
    from xrd_tools.core.containers import IntegrationResult2D
    if rad is None:
        rad = np.linspace(0.5, 5.0, nq, dtype=np.float32)
    return IntegrationResult2D(
        radial=np.asarray(rad, dtype=np.float32),
        azimuthal=np.linspace(-180, 180, nchi, endpoint=False).astype(np.float32),
        intensity=np.full((len(rad), nchi), float(val), dtype=np.float32),
        sigma=None, unit="q_A^-1")


def _make_host(scan, data_2d, data_1d=None):
    from xdart.gui.tabs.static_scan.display_data import DisplayDataMixin
    h = DisplayDataMixin.__new__(DisplayDataMixin)
    h.scan = scan
    h.data_1d = data_1d if data_1d is not None else {}
    h.data_2d = data_2d
    h.data_lock = threading.RLock()
    h.normChannel = None
    h.idxs_2d = []
    h.idxs_1d = []
    h.ui = SimpleNamespace(
        normChannel=SimpleNamespace(currentData=lambda: None, currentText=lambda: ""),
        plotUnit=SimpleNamespace(currentIndex=lambda: 0, currentText=lambda: ""),
        slice=SimpleNamespace(isChecked=lambda: False))
    h.viewer_mode = None
    h.publication_store = None
    h._async_hydration_enabled = False
    return h


def _build_scan_on_disk(tmp_path, N, nq=8, nchi=6):
    from xdart.modules.ewald import LiveScan
    from xdart.modules.ewald.frame import LiveFrame
    nxs = str(tmp_path / "scan.nxs")
    scan = LiveScan(data_file=nxs)
    scan.skip_2d = False
    for i in range(N):
        fr = LiveFrame(idx=i)
        fr.int_1d = _ir1d(i, nq)
        fr.int_2d = _ir2d(i, nq=nq, nchi=nchi)
        fr.scan_info = {}
        fr.source_file = ""
        fr.source_frame_idx = 0
        scan.add_frame(frame=fr, calculate=False, update=True,
                       get_sd=True, batch_save=True)
    scan._save_to_nexus()
    reloaded = LiveScan(data_file=nxs)
    reloaded.load_from_h5()
    return reloaded


class TestCrossFrame2DHydration:
    def test_get_int_2d_no_normalize_returns_source_view(self):
        scan = SimpleNamespace(frames=SimpleNamespace(index=[0]))
        host = _make_host(scan, data_2d={})
        result = _ir2d(4.0, nq=4, nchi=3)

        data = host.get_int_2d(result, normalize=False)

        assert np.shares_memory(data, result.intensity)
        np.testing.assert_allclose(data, result.intensity)

    def test_store_backed_scan_reads_do_not_need_legacy_mirrors(self):
        from xdart.modules.frame_publication import (
            PublicationStore,
            publication_from_live_frame,
        )

        scan = SimpleNamespace(
            frames=SimpleNamespace(index=[0, 1]),
            mask_sentinel=False,
            gi=False,
            bai_1d_args={},
            bai_2d_args={},
        )
        store = PublicationStore(max_heavy_items=None)
        for idx, value in enumerate((1.0, 2.0)):
            frame = SimpleNamespace(
                idx=idx,
                int_1d=_ir1d(value, nq=4),
                int_2d=_ir2d(value, nq=4, nchi=3),
                map_raw=np.full((3, 4), value * 10.0),
                bg_raw=0,
                mask=None,
                gi_2d={},
                thumbnail=None,
                scan_info={},
                source_file=f"frame_{idx}.tif",
                source_frame_idx=idx,
            )
            store.upsert(publication_from_live_frame(frame, include_raw=True))
        host = _make_host(scan, data_2d={}, data_1d={})
        host.publication_store = store

        raw = host.get_frames_map_raw([0, 1], require_all=True)
        cake, _x, _y = host.get_frames_int_2d([0, 1], require_all=True)
        curve, x = host.get_frames_int_1d([0, 1], rv="average")

        np.testing.assert_allclose(raw, 15.0)
        np.testing.assert_allclose(cake, 1.5)
        np.testing.assert_allclose(curve, 1.5)
        np.testing.assert_allclose(x, _ir1d(0, nq=4).radial)

    def test_int_2d_hydrates_evicted_frames(self, tmp_path):
        N = 30
        scan = _build_scan_on_disk(tmp_path, N)
        # data_2d EMPTY -> every selected frame is a cache miss -> must hydrate.
        result, _x, _y = _make_host(scan, data_2d={}).get_frames_int_2d(list(range(N)))
        assert result is not None, "cross-frame 2D returned None (hydration failed)"
        np.testing.assert_allclose(result, (N - 1) / 2.0, atol=1e-4)

    def test_int_2d_hydrated_equals_cached(self, tmp_path):
        N = 30
        scan = _build_scan_on_disk(tmp_path, N)
        hydrated, _, _ = _make_host(scan, {}).get_frames_int_2d(list(range(N)))
        # Production populates data_1d + data_2d together, so the cached path
        # normalizes via the frame's scan_info (identity here) — matching what
        # hydration does with the reloaded LiveFrame.
        full = {i: {'int_2d': scan.frames[i].int_2d, 'gi_2d': {}} for i in range(N)}
        d1d = {i: SimpleNamespace(scan_info={}, thumbnail=None) for i in range(N)}
        cached, _, _ = _make_host(scan, full, d1d).get_frames_int_2d(list(range(N)))
        np.testing.assert_allclose(hydrated, cached)

    def test_int_2d_axis_guard_excludes_mismatched(self, tmp_path):
        scan = _build_scan_on_disk(tmp_path, 2)   # real scan for get_xydata
        rad_a = np.linspace(0.5, 5.0, 8, dtype=np.float32)
        rad_b = np.linspace(1.0, 9.0, 8, dtype=np.float32)   # same shape, diff grid
        data_2d = {
            0: {'int_2d': _ir2d(1.0, rad=rad_a), 'gi_2d': {}},
            1: {'int_2d': _ir2d(99.0, rad=rad_b), 'gi_2d': {}},
        }
        data_1d = {0: SimpleNamespace(scan_info={}, thumbnail=None),
                   1: SimpleNamespace(scan_info={}, thumbnail=None)}
        result, _x, _y = _make_host(scan, data_2d, data_1d).get_frames_int_2d([0, 1])
        # Frame 1 (different q grid) excluded -> result is frame 0's value (1.0),
        # NOT the silent misaligned mean of 1 and 99.
        np.testing.assert_allclose(result, 1.0)

    def test_cache_hit_unchanged_when_all_present(self, tmp_path):
        scan = _build_scan_on_disk(tmp_path, 5)
        full = {i: {'int_2d': scan.frames[i].int_2d, 'gi_2d': {}} for i in range(5)}
        d1d = {i: SimpleNamespace(scan_info={}, thumbnail=None) for i in range(5)}
        result, _, _ = _make_host(scan, full, d1d).get_frames_int_2d(list(range(5)))
        np.testing.assert_allclose(result, 2.0, atol=1e-4)   # mean(0..4)


class TestHydrationGuardDuringRun:
    """Freeze fix (path #2): while a run is writing the .nxs, the GUI-thread
    readers must NOT open the file (the catch_h5py_file retry-storm under
    file_lock = multi-minute beachball).  They serve cache misses from the
    writer's resident in-memory frames only and degrade gracefully to "nothing
    usable" for evicted frames; the disk hydration runs only when idle."""

    @staticmethod
    def _spy_opens(monkeypatch):
        import xdart.modules.ewald.frame_series as fs
        opens = []
        real = fs.catch

        def spy(fname, mode='r', *a, **k):
            opens.append(fname)
            return real(fname, mode, *a, **k)
        monkeypatch.setattr(fs, 'catch', spy)
        return opens

    def test_no_disk_read_during_run_2d(self, tmp_path, monkeypatch):
        opens = self._spy_opens(monkeypatch)
        N = 30
        scan = _build_scan_on_disk(tmp_path, N)   # reloaded -> _in_memory empty
        opens.clear()
        host = _make_host(scan, data_2d={})
        host._processing_active = True            # a run is writing the .nxs
        result, _x, _y = host.get_frames_int_2d(list(range(N)))
        # No .nxs opened on the calling (GUI) thread, and an all-evicted
        # selection degrades to "nothing usable" instead of blocking or raising.
        assert opens == [], f"hydration opened the .nxs during a run: {opens}"
        assert result is None

    def test_run_serves_resident_skips_evicted_2d(self, tmp_path, monkeypatch):
        opens = self._spy_opens(monkeypatch)
        N = 30
        scan = _build_scan_on_disk(tmp_path, N)
        scan.frames._in_memory[5] = scan.frames[5]   # make frame 5 resident; 25 stays evicted
        opens.clear()
        host = _make_host(scan, data_2d={})
        host._processing_active = True
        result, _x, _y = host.get_frames_int_2d([5, 25])
        assert opens == [], f"served from disk during a run: {opens}"
        # Resident frame 5 displays; evicted 25 is skipped (NOT read from disk) ->
        # result is frame 5's value, proving no disk fallback fired.
        np.testing.assert_allclose(result, 5.0, atol=1e-4)

    def test_run_1d_degrades_gracefully(self, tmp_path, monkeypatch):
        opens = self._spy_opens(monkeypatch)
        scan = _build_scan_on_disk(tmp_path, 30)
        opens.clear()
        host = _make_host(scan, data_2d={})
        host._processing_active = True
        ydata, xdata = host.get_frames_int_1d([10, 20], rv='average')
        assert opens == []                            # 1D reader also gated
        assert ydata is None                          # graceful skip, no raise

    def test_idle_still_hydrates_after_run(self, tmp_path):
        # The guard must NOT break the idle path the f51db68 fix targets
        # (post-run reload / whole-scan Set-Bkg on a finished file).
        N = 30
        scan = _build_scan_on_disk(tmp_path, N)
        host = _make_host(scan, data_2d={})
        host._processing_active = False               # idle -> full disk hydration
        result, _, _ = host.get_frames_int_2d(list(range(N)))
        np.testing.assert_allclose(result, (N - 1) / 2.0, atol=1e-4)


class TestCrossFrame1DHydration:
    def test_int_1d_hydrates_evicted_frames(self, tmp_path):
        # 1D sibling of the same bug (data_1d is also a bounded cache): a 1D
        # average over a selection larger than the window must hydrate from disk,
        # not silently average only the cached frames (Set-Bkg consistency).
        N = 40
        scan = _build_scan_on_disk(tmp_path, N)
        host = _make_host(scan, data_2d={})           # data_1d empty too
        ydata, xdata = host.get_frames_int_1d(list(range(N)), rv='average')
        assert ydata is not None, "1D cross-frame returned None (hydration failed)"
        np.testing.assert_allclose(ydata, (N - 1) / 2.0, atol=1e-4)  # mean(0..N-1)


def test_int_2d_require_all_distinguishes_partial_from_no_2d(tmp_path):
    """#6 (Set-Bkg refuse contract): require_all=True returns None when not every
    selected frame contributes (partial coverage), while the default subset path
    still returns the covered average -- so setBkg can tell PARTIAL coverage
    (refuse: a partial average is a wrong background) from a 1D-only scan (no 2D,
    None on both)."""
    N = 5
    scan = _build_scan_on_disk(tmp_path, N)

    host = _make_host(scan, data_2d={})
    # Every frame hydrates from disk -> require_all=True yields the full average.
    assert host.get_frames_int_2d(list(range(N)), require_all=True)[0] is not None

    # A selection including a non-existent frame -> partial coverage.
    host2 = _make_host(scan, data_2d={})
    assert host2.get_frames_int_2d([0, 1, 999])[0] is not None              # subset average exists
    assert host2.get_frames_int_2d([0, 1, 999], require_all=True)[0] is None  # not all covered -> None


class TestSetBkgRawBlockingRead:
    """Set-Bkg whole-scan raw aggregation must block-and-read an evicted frame
    rather than serve the async path (which returns nothing for an evicted frame
    -> require_all=True None -> a silent bkg_map_raw = 0)."""

    @staticmethod
    def _host_async_with_evicted():
        # frame 0 resident (raw=ones); frame 1 EVICTED (only a blocking disk
        # read finds it).  No thumbnails -> the hydrate branch fires.
        host = _make_host(
            scan=SimpleNamespace(frames=SimpleNamespace(index=[0, 1]),
                                 mask_sentinel=False),
            data_2d={0: {'map_raw': np.ones((4, 4)), 'bg_raw': 0}})
        host._async_hydration_enabled = True
        host._processing_active = False
        host.normalize = lambda data, info: data
        calls = []
        evicted = SimpleNamespace(map_raw=np.full((4, 4), 3.0), bg_raw=0,
                                  thumbnail=None, scan_info={},
                                  free_raw=lambda: None)

        def fake_hydrate(idx, *, allow_blocking_read=True):
            calls.append((int(idx), allow_blocking_read))
            if int(idx) == 1 and allow_blocking_read:
                return evicted
            return None

        host._hydrate_frame_from_disk = fake_hydrate
        host._request_frame_hydration = lambda idx: None
        return host, calls

    def test_setbkg_blocks_and_reads_evicted_frame(self):
        host, calls = self._host_async_with_evicted()
        data = host.get_frames_map_raw([0, 1], require_all=True,
                                       allow_blocking_read=True)
        assert (1, True) in calls               # forced a blocking read
        assert data is not None                  # both frames covered...
        np.testing.assert_allclose(data, 2.0, atol=1e-6)  # (ones + threes)/2

    def test_default_render_path_stays_async_nonblocking(self):
        # Regression guard: without the override the live render path must NOT
        # block on the evicted frame -> require_all None + a background request.
        host, calls = self._host_async_with_evicted()
        queued = []
        host._request_frame_hydration = lambda idx: queued.append(int(idx))
        out = host.get_frames_map_raw([0, 1], require_all=True)   # no override
        assert (1, False) in calls               # served the async helper
        assert queued == [1]                      # evicted frame queued off-thread
        assert out is None                        # async path can't complete inline
