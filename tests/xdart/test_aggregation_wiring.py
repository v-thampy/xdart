# -*- coding: utf-8 -*-
"""Wiring of the whole-scan aggregate into the live display (Step 7b A1b-2c).

The user-reported bug: a live Int 2D Overall cake goes BLANK on Stop for a scan
longer than the bounded store (>64 frames) — §2.C correctly refuses to average a
wrong (store-resident-only) subset, but nothing filled that blank.  These tests
lock the fill: the widget computes the whole-scan aggregate (disk ⊕ tail) and the
cake adapter routes the §2.C blank to it instead of returning None.
"""

from __future__ import annotations

from threading import RLock
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

NQ, NCHI = 6, 4


def _split_scan_2d(tmp_path, *, n=30, cap=8):
    """A REAL LiveScan with a 2D stack, longer than the in-memory cap so the
    store can't hold it all (frames are flushed to disk; values 1..n)."""
    from xdart.modules.ewald import LiveScan
    from xdart.modules.ewald.frame import LiveFrame
    from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
    q = np.linspace(0.5, 3.0, NQ, dtype=np.float32)
    chi = np.linspace(-90.0, 90.0, NCHI, dtype=np.float32)
    scan = LiveScan(data_file=str(tmp_path / "cake.nxs"))
    scan.skip_2d = False
    scan.frames._in_memory_cap = cap

    def mk(i):
        fr = LiveFrame(idx=i)
        fr.int_1d = IntegrationResult1D(
            radial=q, intensity=np.full(NQ, float(i + 1), np.float32),
            sigma=np.ones(NQ, np.float32), unit="q_A^-1")
        fr.int_2d = IntegrationResult2D(            # (radial, azimuthal)=(nq, nchi)
            radial=q, azimuthal=chi,
            intensity=np.full((NQ, NCHI), float(i + 1), np.float32),
            unit="q_A^-1", azimuthal_unit="chi_deg")
        fr.scan_info = {"i0": float(i + 1)}
        fr.source_file = ""
        fr.source_frame_idx = 0
        fr.skip_map_raw = True
        return fr

    for i in range(n):
        scan.add_frame(frame=mk(i), calculate=False, update=True,
                       get_sd=True, batch_save=True)
        if (i + 1) % 10 == 0:
            scan._save_to_nexus()
    scan._save_to_nexus()
    return scan, q, chi


# ── widget method: _whole_scan_aggregate ──────────────────────────────────────
# Called unbound with a duck ``self`` — the method only reads scan /
# display_generation / _agg_cache / get_normChannel / _async_hydration_enabled,
# so the sync path needs no QApplication.

def _duck_widget(scan, **over):
    from types import SimpleNamespace
    d = SimpleNamespace(scan=scan, display_generation=1, _agg_cache={},
                        _async_hydration_enabled=False,
                        get_normChannel=lambda: None)
    for k, v in over.items():
        setattr(d, k, v)
    return d


def _call_whole_scan_aggregate(duck, **kw):
    from xdart.gui.tabs.static_scan.display_frame_widget import displayFrameWidget
    return displayFrameWidget._whole_scan_aggregate(duck, **kw)


def test_widget_whole_scan_aggregate_2d_sync(tmp_path):
    scan, q, chi = _split_scan_2d(tmp_path, n=30)
    duck = _duck_widget(scan)
    agg = _call_whole_scan_aggregate(duck, dim="2d", method="average")
    assert agg is not None
    assert agg.intensity.shape == (NCHI, NQ)     # disk/get_2d convention
    np.testing.assert_allclose(agg.intensity, np.mean(np.arange(1, 31)))   # 15.5


def test_widget_whole_scan_aggregate_caches_per_generation(tmp_path):
    scan, _, _ = _split_scan_2d(tmp_path, n=12)
    duck = _duck_widget(scan)
    _call_whole_scan_aggregate(duck, dim="2d", method="average")
    key = ("2d", "average", None)
    assert key in duck._agg_cache and duck._agg_cache[key][0] == duck.display_generation


def test_widget_async_aggregate_none_is_retryable(tmp_path):
    from xdart.gui.tabs.static_scan.display_frame_widget import displayFrameWidget
    scan, _, _ = _split_scan_2d(tmp_path, n=12)
    calls = []

    class FakeWorker:
        def request(self, *args):
            calls.append(args)

    duck = _duck_widget(
        scan,
        _async_hydration_enabled=True,
        _agg_pending=set(),
    )
    duck._ensure_aggregation_worker = lambda: FakeWorker()

    assert _call_whole_scan_aggregate(duck, dim="2d", method="average") is None
    assert len(calls) == 1
    # Re-render before the worker replies must not enqueue the same work again.
    assert _call_whole_scan_aggregate(duck, dim="2d", method="average") is None
    assert len(calls) == 1

    key = ("2d", "average", None)
    displayFrameWidget._on_aggregated(duck, key, duck.display_generation, None)
    assert key not in duck._agg_cache

    # A later scan/display update at the same generation can retry; the earlier
    # None was "not ready", not a terminal empty aggregate.
    assert _call_whole_scan_aggregate(duck, dim="2d", method="average") is None
    assert len(calls) == 2


def test_active_aggregate_pending_survives_selection_generation_bumps(tmp_path):
    scan, _, _ = _split_scan_2d(tmp_path, n=12)
    calls = []

    class FakeWorker:
        def request(self, *args):
            calls.append(args)

    duck = _duck_widget(
        scan,
        _async_hydration_enabled=True,
        _processing_active=True,
        _aggregate_live_scan=scan,
        _agg_pending=set(),
        _agg_generation=0,
        _agg_signature_by_key={},
    )
    duck._ensure_aggregation_worker = lambda: FakeWorker()

    assert _call_whole_scan_aggregate(duck, dim="2d", method="average") is None
    duck.display_generation += 1
    assert _call_whole_scan_aggregate(duck, dim="2d", method="average") is None

    assert len(calls) == 1
    assert calls[0][1] == 1


def test_active_aggregate_cache_invalidates_only_on_flush_signature(
        tmp_path, monkeypatch):
    from xrd_tools.io.aggregate import Aggregated2D
    import xdart.modules.scan_aggregate as scan_aggregate_mod

    scan, q, chi = _split_scan_2d(tmp_path, n=12)
    calls = []

    def fake_aggregate(scan_arg, *, method, norm_channel=None):
        calls.append((scan_arg, method, norm_channel))
        return Aggregated2D(
            q=q, chi=chi,
            intensity=np.full((chi.size, q.size), float(len(calls))),
            q_unit="q_A^-1", chi_unit="chi_deg", n_frames=12,
        )

    monkeypatch.setattr(
        scan_aggregate_mod, "whole_scan_aggregate_2d", fake_aggregate)
    duck = _duck_widget(
        scan,
        _processing_active=True,
        _aggregate_live_scan=scan,
        _agg_generation=0,
        _agg_signature_by_key={},
    )

    first = _call_whole_scan_aggregate(duck, dim="2d", method="average")
    duck.display_generation += 1
    second = _call_whole_scan_aggregate(duck, dim="2d", method="average")
    scan.frames.mark_persisted([999])
    third = _call_whole_scan_aggregate(duck, dim="2d", method="average")

    assert len(calls) == 2
    assert first.intensity[0, 0] == 1.0
    assert second.intensity[0, 0] == 1.0
    assert third.intensity[0, 0] == 2.0


def test_active_aggregate_uses_worker_scan_for_unflushed_tail(tmp_path, monkeypatch):
    from xrd_tools.io.aggregate import Aggregated2D
    import xdart.modules.scan_aggregate as scan_aggregate_mod

    display_scan, q, chi = _split_scan_2d(tmp_path, n=12)
    live_scan = SimpleNamespace(
        data_file=display_scan.data_file,
        frames=display_scan.frames,
        gi=getattr(display_scan, "gi", False),
    )
    seen = []

    def fake_aggregate(scan_arg, *, method, norm_channel=None):
        seen.append(scan_arg)
        return Aggregated2D(
            q=q, chi=chi, intensity=np.ones((chi.size, q.size)),
            q_unit="q_A^-1", chi_unit="chi_deg", n_frames=12,
        )

    monkeypatch.setattr(
        scan_aggregate_mod, "whole_scan_aggregate_2d", fake_aggregate)
    duck = _duck_widget(
        display_scan,
        _processing_active=True,
        _aggregate_live_scan=live_scan,
        _agg_generation=0,
        _agg_signature_by_key={},
    )

    assert _call_whole_scan_aggregate(duck, dim="2d", method="average") is not None
    assert seen == [live_scan]


def test_sum_average_state_snapshot_does_not_reenqueue_full_2d_hydration():
    from xdart.gui.tabs.static_scan.display_controllers import ScanDisplayController
    from xdart.gui.tabs.static_scan.display_logic import Mode

    queued = []
    def request(label, *, purpose="full"):
        queued.append((label, purpose))

    pub = SimpleNamespace(
        view=SimpleNamespace(
            has_1d=True,
            has_2d=False,
            intensity_1d=np.ones(3),
            intensity_2d=None,
            raw=None,
            thumbnail=None,
        )
    )
    store = SimpleNamespace(get=lambda _label: pub)
    widget = SimpleNamespace(
        frame_ids=["0", "1", "2"],
        viewer_mode=None,
        publication_store=store,
        viewer_rows_1d={},
        viewer_rows_2d={},
        overlaid_idxs=[],
        display_generation=1,
        scan=SimpleNamespace(
            scan_lock=RLock(),
            frames=SimpleNamespace(index=[0, 1, 2]),
            gi=False,
        ),
        ui=SimpleNamespace(
            plotMethod=SimpleNamespace(currentText=lambda: "Average"),
        ),
        _whole_scan_aggregate=lambda *, dim, method: None,
        _request_frame_hydration=request,
    )

    controller = ScanDisplayController()

    first = controller.compute_state(widget, Mode.INT_2D)
    second = controller.compute_state(widget, Mode.INT_2D)

    assert first.overall is True
    assert second.overall is True
    assert queued == []


def test_explicit_sum_subset_missing_frames_hydrates_or_refuses():
    from xdart.gui.tabs.static_scan.display_logic import Mode
    from xdart.gui.tabs.static_scan.display_publication import (
        PublicationDisplayAdapter)
    from xdart.modules.frame_publication import (
        PublicationStore, publication_from_live_frame)
    from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D

    x = np.linspace(0.5, 5.0, 6, dtype=np.float32)

    def _frame(i):
        return SimpleNamespace(
            idx=i,
            int_1d=IntegrationResult1D(
                radial=x, intensity=np.full(x.shape, float(i), dtype=np.float32),
                sigma=np.ones_like(x), unit="q_A^-1"),
            int_2d=IntegrationResult2D(
                radial=np.linspace(0.5, 5.0, 4, dtype=np.float32),
                azimuthal=np.linspace(-90.0, 90.0, 3, dtype=np.float32),
                intensity=np.full((4, 3), float(i), dtype=np.float32),
                unit="q_A^-1", azimuthal_unit="chi_deg"),
            map_raw=None, mask=None, gi=False, gi_2d={},
            thumbnail=None, bg_raw=0, scan_info={}, source_file=f"f{i}.tif",
            source_frame_idx=i,
        )

    state = SimpleNamespace(
        mode=Mode.INT_1D,
        method="Sum",
        overall=False,
        selected_ids=(1, 2, 3),
        render_ids=(1, 2, 3),
    )
    widget = SimpleNamespace(
        _async_hydration_enabled=True,
        _plot_axis_info=[{"source": "1d"}],
        normalize=lambda data, _metadata: data,
        scan=SimpleNamespace(name="scan", gi=False),
        ui=SimpleNamespace(
            plotUnit=SimpleNamespace(currentIndex=lambda: 0,
                                     currentText=lambda: "Q (Å⁻¹)"),
            slice=SimpleNamespace(isChecked=lambda: False),
        ),
    )

    queued = []
    widget._request_frame_hydration = (
        lambda label, *, purpose="full": queued.append((label, purpose)))
    live_store = PublicationStore(max_heavy_items=1)
    for label in state.selected_ids:
        live_store.upsert(publication_from_live_frame(_frame(label)))

    payload = PublicationDisplayAdapter(
        live_store, widget=widget, labels=state.selected_ids
    ).plot_payload(state)

    assert payload is not None
    assert payload.traces == ()
    assert queued == [(1, "1d"), (2, "1d")]

    calls = []
    sync_store = PublicationStore(max_heavy_items=None)

    def hydrate(labels):
        calls.append(tuple(labels))
        return [publication_from_live_frame(_frame(label)) for label in labels]

    sync_store.set_1d_hydrator(hydrate)
    widget._async_hydration_enabled = False
    queued.clear()

    payload = PublicationDisplayAdapter(
        sync_store, widget=widget, labels=state.selected_ids
    ).plot_payload(state)

    assert calls == [(1, 2, 3)]
    assert queued == []
    assert payload is not None
    assert len(payload.traces) == 3


def test_widget_whole_scan_aggregate_allows_primary_gi(tmp_path):
    scan, _, _ = _split_scan_2d(tmp_path, n=12)
    scan.gi = True
    scan.bai_2d_args = {"gi_mode_2d": "qip_qoop"}
    scan.gi_config = {"gi_mode_2d": "qip_qoop"}
    duck = _duck_widget(scan)
    agg = _call_whole_scan_aggregate(duck, dim="2d", method="average")
    assert agg is not None
    np.testing.assert_allclose(agg.intensity, np.mean(np.arange(1, 13)))


def test_widget_whole_scan_aggregate_defers_for_nonprimary_gi(tmp_path):
    scan, _, _ = _split_scan_2d(tmp_path, n=12)
    scan.gi = True
    scan.bai_2d_args = {"gi_mode_2d": "q_chi"}          # currently displayed
    scan.gi_config = {"gi_mode_2d": "qip_qoop"}         # primary on disk
    duck = _duck_widget(scan)
    assert _call_whole_scan_aggregate(duck, dim="2d", method="average") is None


def test_widget_whole_scan_aggregate_defers_for_gi_without_gi_config(tmp_path):
    # FAIL-CLOSED: a GI scan whose gi_config does not record the primary mode
    # (e.g. an older-format .nxs reloaded) must DEFER the whole-scan aggregate,
    # not default primary=displayed (which would always pass the gate and defeat
    # the anti-truncation protection when the user switches GI mode).
    scan, _, _ = _split_scan_2d(tmp_path, n=12)
    scan.gi = True
    scan.bai_2d_args = {"gi_mode_2d": "q_chi"}
    scan.gi_config = {}                                  # no recorded primary mode
    duck = _duck_widget(scan)
    assert _call_whole_scan_aggregate(duck, dim="2d", method="average") is None


@pytest.fixture
def widget(qapp):
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    w = staticWidget()
    try:
        yield w
    finally:
        try:
            h5v = w.h5viewer
            h5v.cancel_pending_loads()
            ft = getattr(h5v, "file_thread", None)
            if ft is not None:
                ft.queue.put(None)
                ft.wait(2000)
            pool = getattr(h5v, "_h5pool", None)
            if pool is not None:
                pool.close_all()
        except Exception:
            pass
        try:
            w.close()
        except Exception:
            pass
        qapp.processEvents()


@pytest.mark.gui
def test_real_widget_overall_aggregate_uses_disk_when_store_evicted(
    widget, tmp_path
):
    # §0.2 e2e gap: drive the real displayFrameWidget.update() path with more
    # frames than PublicationStore keeps heavy.  Old mirrors are empty, so the
    # only correct result is the disk-backed whole-scan aggregate, not a resident
    # subset and not a blank.
    from xdart.modules.frame_publication import publication_from_frame_view
    from xrd_tools.core import FrameView
    from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D

    n = 70
    scan, q, chi = _split_scan_2d(tmp_path, n=n, cap=8)
    w = widget
    df = w.displayframe
    df.scan = scan
    df.viewer_mode = None
    df._async_hydration_enabled = False
    df.ui.plotMethod.setCurrentText("Average")
    df.ui.plotUnit.setCurrentIndex(0)

    with w.data_lock:
        w.viewer_rows_1d.clear()
        w.viewer_rows_2d.clear()
    w.publication_store.clear()
    for i in range(n):
        r1 = IntegrationResult1D(
            radial=q,
            intensity=np.full(q.shape, float(i + 1), dtype=np.float32),
            sigma=None,
            unit="q_A^-1",
        )
        r2 = IntegrationResult2D(
            radial=q,
            azimuthal=chi,
            intensity=np.full((q.size, chi.size), float(i + 1), dtype=np.float32),
            unit="q_A^-1",
            azimuthal_unit="chi_deg",
        )
        view = FrameView.from_results(label=i, result_1d=r1, result_2d=r2)
        w.publication_store.upsert(publication_from_frame_view(view))

    assert len(w.publication_store.labels()) == n
    assert sum(
        1
        for i in range(n)
        if (pub := w.publication_store.get(i)) is not None and pub.view.has_1d
    ) < n

    w.frame_ids[:] = [str(i) for i in range(n)]
    df.frame_ids = list(w.frame_ids)
    df.update()

    expected = np.mean(np.arange(1, n + 1))
    assert df.binned_data is not None
    np.testing.assert_allclose(df.binned_data[0], expected)
    x, y = df.plot_data
    np.testing.assert_allclose(x, q)
    np.testing.assert_allclose(np.asarray(y), expected)


def _evict_overall_store(w, q, chi, n):
    """Clear the legacy mirrors + store, then upsert ``n`` Overall frame
    publications whose heavy rows the bounded store thins — so a subsequent
    Overall render can only be satisfied from the on-disk whole-scan aggregate.
    Returns after pointing ``w.frame_ids`` at the full range."""
    from xdart.modules.frame_publication import publication_from_frame_view
    from xrd_tools.core import FrameView
    from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D

    with w.data_lock:
        w.viewer_rows_1d.clear()
        w.viewer_rows_2d.clear()
    w.publication_store.clear()
    for i in range(n):
        r1 = IntegrationResult1D(
            radial=q, intensity=np.full(q.shape, float(i + 1), dtype=np.float32),
            sigma=None, unit="q_A^-1")
        r2 = IntegrationResult2D(
            radial=q, azimuthal=chi,
            intensity=np.full((q.size, chi.size), float(i + 1), dtype=np.float32),
            unit="q_A^-1", azimuthal_unit="chi_deg")
        view = FrameView.from_results(label=i, result_1d=r1, result_2d=r2)
        w.publication_store.upsert(publication_from_frame_view(view))
    # Eviction actually happened: not every row still carries its heavy 1D payload.
    assert sum(1 for i in range(n)
               if (p := w.publication_store.get(i)) is not None
               and p.view.has_1d) < n
    w.frame_ids[:] = [str(i) for i in range(n)]


@pytest.mark.gui
def test_real_widget_gi_primary_mode_serves_disk_aggregate(widget, tmp_path):
    # A-3a (GI render-through PAIR — primary half): a GI scan whose DISPLAYED
    # 1D/2D modes match the primary modes recorded in gi_config serves the Overall
    # cake + 1D from the disk-tail whole-scan aggregate (store heavy rows evicted,
    # legacy mirrors empty) through the REAL df.update() — not just the unbound
    # _whole_scan_aggregate gate (test_widget_whole_scan_aggregate_allows_primary_gi).
    n = 70
    scan, q, chi = _split_scan_2d(tmp_path, n=n, cap=8)
    scan.gi = True
    scan.bai_1d_args = {"gi_mode_1d": "q_total"}
    scan.bai_2d_args = {"gi_mode_2d": "qip_qoop"}
    scan.gi_config = {"gi_mode_1d": "q_total", "gi_mode_2d": "qip_qoop"}  # == displayed
    w = widget
    df = w.displayframe
    df.scan = scan
    df.viewer_mode = None
    df._async_hydration_enabled = False
    df.ui.plotMethod.setCurrentText("Average")
    df.ui.plotUnit.setCurrentIndex(0)

    _evict_overall_store(w, q, chi, n)
    df.frame_ids = list(w.frame_ids)
    df.update()

    expected = np.mean(np.arange(1, n + 1))
    # binned_data inits to (None, None); assert the cake datum is actually drawn,
    # not merely that the tuple exists.
    assert df.binned_data is not None and df.binned_data[0] is not None
    np.testing.assert_allclose(df.binned_data[0], expected)
    x, y = df.plot_data
    np.testing.assert_allclose(np.asarray(y), expected)


@pytest.mark.gui
def test_real_widget_gi_nonprimary_mode_blanks_instead_of_partial(widget, tmp_path):
    # A-3b (GI render-through PAIR — non-primary half): a GI scan whose displayed
    # 2D mode does NOT match the primary mode in gi_config must BLANK the Overall
    # cake (and the 1D, whose gi_mode_1d is unrecorded -> fail-closed) rather than
    # render a wrong/partial aggregate from a bounded resident subset.  Drives the
    # REAL df.update(); the adapter-level analogue is
    # test_cake_image_blocks_nonprimary_gi_instead_of_falling_back.
    n = 70
    scan, q, chi = _split_scan_2d(tmp_path, n=n, cap=8)
    scan.gi = True
    scan.bai_2d_args = {"gi_mode_2d": "q_chi"}          # displayed (non-primary)
    scan.gi_config = {"gi_mode_2d": "qip_qoop"}         # primary on disk (mismatch)
    w = widget
    df = w.displayframe
    df.scan = scan
    df.viewer_mode = None
    df._async_hydration_enabled = False
    df.ui.plotMethod.setCurrentText("Average")
    df.ui.plotUnit.setCurrentIndex(0)

    _evict_overall_store(w, q, chi, n)
    df.frame_ids = list(w.frame_ids)
    df.update()

    # Non-primary GI Overall: no wrong/partial aggregate is rendered.
    assert df.binned_data is None
    _x, y = df.plot_data
    assert np.asarray(y).size == 0


@pytest.mark.gui
def test_real_widget_setbkg_hydrates_evicted_subset_from_disk(widget, tmp_path):
    # A-1: per-frame disk hydration of a NON-Overall subset whose rows are evicted
    # from the store (legacy mirrors empty).  get_frames_int_2d/1d feed ONLY
    # Set-Bkg (display_data.py), so Set-Bkg is the real consumer of that hydration
    # path — drive the REAL displayFrameWidget.setBkg() and prove the 1D/2D
    # background is hydrated from disk (the correct subset average), not refused as
    # partial-coverage and not left blank.  (The mixin-level analogue is
    # test_display_cross_frame_2d.py; the normal subset *display* deliberately
    # blanks a subset aggregate — test_cake_image_blanks_non_overall_eviction.)
    n = 70
    scan, q, chi = _split_scan_2d(tmp_path, n=n, cap=8)
    w = widget
    df = w.displayframe
    df.scan = scan
    df.viewer_mode = None
    df._async_hydration_enabled = False

    _evict_overall_store(w, q, chi, n)            # clears mirrors + thins the store
    # NON-Overall subset of the FIFO tier-1-EVICTED labels (the store thins oldest
    # first), so setBkg's get_frames_int_2d/1d MUST hydrate them from disk — not
    # serve them resident.  Lock that precondition: if the eviction policy ever
    # changes so these rows stay resident, fail loudly here instead of letting the
    # test silently go vacuous (the adversarial-review finding).
    subset = [0, 1, 2]
    for i in subset:
        p = w.publication_store.get(i)
        assert p is not None and not p.view.has_2d and not p.view.has_1d, (
            f"precondition: frame {i} must be heavy-evicted so setBkg hydrates "
            f"from disk (store cap >= n would make this test vacuous)")
    df.frame_ids = [str(i) for i in subset]
    df.overall = False
    df.idxs = list(subset)
    df.ui.setBkg.setText("Set BG")

    df.setBkg()

    expected = float(np.mean([i + 1 for i in subset]))   # mean(1, 2, 3) = 2.0
    assert df.bkg_2d is not None                  # full-coverage disk hydrate, not refused
    np.testing.assert_allclose(df.bkg_2d, expected)
    assert df.bkg_1d is not None
    np.testing.assert_allclose(np.asarray(df.bkg_1d), expected)


@pytest.mark.gui
def test_idle_overlay_redraw_does_not_reread_when_accumulator_covers_selection(
    widget, tmp_path
):
    # HARD-FREEZE regression: at end-of-scan the mode flips live->idle, so an
    # Overlay/Waterfall whose accumulator ALREADY holds every selected frame must
    # just REDRAW — never re-collect.  Re-collecting on the idle path block-reads
    # the whole scan's 1D from disk on the GUI thread (~1 s for a 651-frame scan
    # past the store cap) on EVERY refresh, which (repeated by autorange / the
    # aggregate worker / the final flush) saturates the GUI thread into a total
    # freeze.  Assert update_plot() issues NO get_frames_int_1d fetch here.
    n = 30
    scan, q, _chi = _split_scan_2d(tmp_path, n=n, cap=8)
    w = widget
    df = w.displayframe
    df.scan = scan
    df.viewer_mode = None
    df._async_hydration_enabled = False
    df.ui.plotMethod.setCurrentText("Waterfall")
    df.ui.plotUnit.setCurrentIndex(0)
    # Simulate live having already accumulated all n rows into the waterfall.
    df.plot_data = [q.astype(float),
                    (np.arange(n, dtype=float)[:, None] * np.ones(q.size))]
    df.frame_names = [f"{scan.name}_{i}" for i in range(n)]
    df.overlaid_idxs = list(range(n))
    df.frame_ids = [str(i) for i in range(n)]
    df.idxs = list(range(n))
    df.idxs_1d = list(range(n))
    df._last_plot_unit = df.ui.plotUnit.currentIndex()
    df._processing_active = False                  # idle: scan ended

    calls = []
    orig = df.get_frames_int_1d
    df.get_frames_int_1d = (
        lambda *a, **k: (calls.append(1), orig(*a, **k))[1])
    try:
        df.update_plot()
    finally:
        df.get_frames_int_1d = orig
    assert calls == []                             # redrew from the accumulator; no disk re-read


@pytest.mark.gui
def test_catchup_overlay_reads_only_missing_not_whole_scan(widget, tmp_path):
    # HARD-FREEZE regression (froze at frame 648 of 651): at end-of-scan catch-up
    # the accumulator already holds most frames; the few not-yet-shown frames must
    # be fetched WITHOUT re-reading the whole scan's 1D from disk on the GUI
    # thread (the multi-second blocking read that repeated and never recovered).
    # Assert the fetch covers ONLY the missing frames, never all N.
    # Post flip-stage-3: live Waterfall renders via the payload accumulator (no disk
    # re-read at all); this guards the LEGACY update_plot fallback, exercised
    # directly (it survives until stage 4 retires it).
    n = 70
    scan, q, chi = _split_scan_2d(tmp_path, n=n, cap=8)
    w = widget
    df = w.displayframe
    df.scan = scan
    df.viewer_mode = None
    df._async_hydration_enabled = False
    df.ui.plotMethod.setCurrentText("Waterfall")
    df.ui.plotUnit.setCurrentIndex(0)
    _evict_overall_store(w, q, chi, n)                    # store populated + thinned past cap
    # Accumulator already holds all but the last 3 frames (live catch-up state).
    df.plot_data = [q.astype(float),
                    (np.arange(n - 3, dtype=float)[:, None] * np.ones(q.size))]
    df.frame_names = [f"{scan.name}_{i}" for i in range(n - 3)]
    df.overlaid_idxs = list(range(n - 3))
    df._last_plot_unit = df.ui.plotUnit.currentIndex()   # unit stable: APPEND, not REBUILD
    df.frame_ids = [str(i) for i in range(n)]
    df._processing_active = False                          # idle: scan ended

    reads = []
    orig = df.get_frames_int_1d
    df.get_frames_int_1d = (
        lambda idxs=None, **k: (
            reads.append(len(list(idxs)) if idxs is not None else -1),
            orig(idxs, **k))[1])
    try:
        df.update_plot()
    finally:
        df.get_frames_int_1d = orig
    assert reads, "expected a fetch of the missing frames"
    assert max(reads) <= 3, f"fetched {max(reads)} frames; must read only the missing few"
    assert max(reads) < n, "must NOT re-read the whole scan from disk"


@pytest.mark.gui
def test_unit_switch_rebuild_never_blocks_gui_thread(widget, tmp_path):
    # HARD-FREEZE regression (Q->2θ unit switch): re-expressing the accumulated
    # waterfall in a new unit triggers update_plot's REBUILD (plus the collect
    # before it).  Neither may pass allow_blocking_read=True (a synchronous
    # whole-scan disk read on the GUI thread that locked the UI for seconds);
    # they route through the async path (None) so the live app reads resident now
    # and the FrameHydrationWorker backfills the rest off-thread.
    n = 70
    scan, q, chi = _split_scan_2d(tmp_path, n=n, cap=8)
    w = widget
    df = w.displayframe
    df.scan = scan
    df.viewer_mode = None
    df.ui.plotMethod.setCurrentText("Waterfall")
    df.ui.plotUnit.setCurrentIndex(0)
    _evict_overall_store(w, q, chi, n)
    df.frame_ids = [str(i) for i in range(n)]
    df.overlaid_idxs = list(range(n))                 # full accumulator (worst case)
    df.plot_data = [q.astype(float),
                    (np.arange(n, dtype=float)[:, None] * np.ones(q.size))]
    df.frame_names = [f"{scan.name}_{i}" for i in range(n)]
    df._processing_active = False                      # idle (scan ended)
    df._last_plot_unit = -999                          # force unit_changed -> REBUILD

    blocking = []
    orig = df.get_frames_int_1d

    def spy(idxs=None, rv="all", *, require_all=False, allow_blocking_read=None):
        blocking.append(allow_blocking_read)
        return orig(idxs, rv, require_all=require_all,
                    allow_blocking_read=allow_blocking_read)
    df.get_frames_int_1d = spy
    try:
        df.update_plot()
    finally:
        df.get_frames_int_1d = orig
    assert blocking, "expected a fetch during the unit-switch rebuild"
    assert all(b is not True for b in blocking), (
        f"GUI-thread blocking read on unit switch: {blocking}")


@pytest.mark.gui
def test_overlay_qtth_unit_switch_transforms_x_without_reread(widget, tmp_path):
    # FREEZE + INCOMPLETENESS regression (Q->2θ on a big overlay/waterfall): the
    # accumulated intensities are unit-invariant, so a display-unit switch
    # re-expresses ONLY the shared x-axis from what is already accumulated -- NO
    # disk re-read (instant, no freeze) and COMPLETE (no cap-store backfill race
    # that previously left the waterfall half-filled).
    import numpy as _np
    n = 20
    scan, q, _chi = _split_scan_2d(tmp_path, n=n, cap=8)
    scan._persisted_wavelength_m = 1.5e-10                # real λ for the transform
    w = widget
    df = w.displayframe
    df.scan = scan
    df.viewer_mode = None
    df.ui.plotMethod.setCurrentText("Waterfall")
    df.ui.plotUnit.clear()
    df.ui.plotUnit.addItems(["Q (Å⁻¹)", "2θ (°)"])
    df._plot_axis_info = [{"source": "1d_2d", "axis": "radial"},
                          {"source": "1d_2d", "axis": "radial"}]
    df.ui.slice.setChecked(False)
    qgrid = q.astype(float)
    yrows = (np.arange(n, dtype=float)[:, None] * np.ones(qgrid.size))
    df.plot_data = [qgrid.copy(), yrows.copy()]
    df.frame_names = [f"{scan.name}_{i}" for i in range(n)]
    df.overlaid_idxs = list(range(n))
    df.ui.plotUnit.setCurrentIndex(1)                     # switch Q -> 2θ

    reads = []
    orig = df.get_frames_int_1d
    df.get_frames_int_1d = lambda *a, **k: (reads.append(1), orig(*a, **k))[1]
    try:
        ok = df._reexpress_overlay_unit(0, 1)            # prev=Q, new=2θ
    finally:
        df.get_frames_int_1d = orig

    assert ok is True                                     # transformed in place
    assert reads == []                                   # no disk re-read at all
    expected = 2 * _np.degrees(_np.arcsin(
        _np.clip(qgrid * 1.5 / (4 * _np.pi), -1, 1)))
    np.testing.assert_allclose(df.plot_data[0], expected)            # exact Q->2θ
    np.testing.assert_allclose(np.asarray(df.plot_data[1]), yrows)   # y untouched

    # NEGATIVE: a χ (source '2d') axis is NOT a pure x-transform -> defer to re-read.
    df._plot_axis_info = [{"source": "1d_2d", "axis": "radial"},
                          {"source": "2d", "axis": "azimuthal"}]
    assert df._reexpress_overlay_unit(0, 1) is False

    # NEGATIVE: no wavelength + family change would mislabel -> defer to re-read.
    df._plot_axis_info = [{"source": "1d_2d", "axis": "radial"},
                          {"source": "1d_2d", "axis": "radial"}]
    scan._persisted_wavelength_m = None
    scan.mg_args = {}
    scan.data_file = ""
    df.plot_data = [qgrid.copy(), yrows.copy()]
    assert df._reexpress_overlay_unit(0, 1) is False


# ── adapter routing: cake_image -> _aggregate_cake_payload ─────────────────────

def _fake_state(*, overall, selected_ids, render_ids=(), method="Average"):
    return SimpleNamespace(
        overall=overall,
        method=method,
        selected_ids=tuple(selected_ids),
        render_ids=tuple(render_ids),
        panel=lambda role: SimpleNamespace(has_data=True, source=None),
    )


def test_cake_image_routes_eviction_to_aggregate():
    from xdart.gui.tabs.static_scan.display_publication import (
        PublicationDisplayAdapter)
    from xrd_tools.io import Aggregated2D
    q = np.linspace(0.5, 3.0, NQ)
    chi = np.linspace(-90.0, 90.0, NCHI)
    agg = Aggregated2D(q=q, chi=chi,
                       intensity=np.full((NCHI, NQ), 15.5),
                       q_unit="q_A^-1", chi_unit="chi_deg", n_frames=30)
    calls = []
    widget = SimpleNamespace(
        bkg_2d=0,
        _whole_scan_aggregate=lambda *, dim, method: calls.append((dim, method)) or agg,
    )
    adapter = PublicationDisplayAdapter(store=None, widget=widget)
    # Overall selection with an evicted frame (label 0 not in the empty store).
    state = _fake_state(overall=True, selected_ids=(0,), render_ids=())
    payload = adapter.cake_image(state)
    assert payload is not None                    # filled, not blanked
    assert payload.image.shape == (NCHI, NQ)
    np.testing.assert_allclose(payload.image, 15.5)
    assert calls == [("2d", "average")]           # routed to the aggregate


def test_cake_image_routes_sum_eviction_to_sum_aggregate():
    from xdart.gui.tabs.static_scan.display_publication import (
        PublicationDisplayAdapter)
    from xrd_tools.io import Aggregated2D
    q = np.linspace(0.5, 3.0, NQ)
    chi = np.linspace(-90.0, 90.0, NCHI)
    agg = Aggregated2D(q=q, chi=chi,
                       intensity=np.full((NCHI, NQ), 31.0),
                       q_unit="q_A^-1", chi_unit="chi_deg", n_frames=30)
    calls = []
    widget = SimpleNamespace(
        bkg_2d=0,
        _whole_scan_aggregate=lambda *, dim, method: calls.append((dim, method)) or agg,
    )
    adapter = PublicationDisplayAdapter(store=None, widget=widget)
    state = _fake_state(
        overall=True, selected_ids=(0,), render_ids=(), method="Sum")

    payload = adapter.cake_image(state)

    assert payload is not None
    np.testing.assert_allclose(payload.image, 31.0)
    assert calls == [("2d", "sum")]


def test_cake_image_blocks_nonprimary_gi_instead_of_falling_back():
    from xdart.gui.tabs.static_scan.display_publication import (
        PublicationDisplayAdapter)

    widget = SimpleNamespace(
        bkg_2d=0,
        scan=SimpleNamespace(),
        _aggregate_display_is_primary=lambda _scan, _dim: False,
        _whole_scan_aggregate=lambda *, dim, method: pytest.fail(
            "non-primary GI aggregate must not read a partial stack"),
    )
    adapter = PublicationDisplayAdapter(store=None, widget=widget)
    state = _fake_state(overall=True, selected_ids=(0,), render_ids=())

    payload = adapter.cake_image(state)

    assert payload is not None
    assert payload.image.size == 0


def test_plot_payload_routes_overall_eviction_to_1d_aggregate():
    from xdart.gui.tabs.static_scan.display_publication import (
        PublicationDisplayAdapter)
    from xrd_tools.io import Aggregated1D
    q = np.linspace(0.5, 3.0, NQ)
    agg = Aggregated1D(
        q=q,
        intensity=np.full(NQ, 15.5),
        q_unit="q_A^-1",
        n_frames=30,
    )
    calls = []
    widget = SimpleNamespace(
        scan=SimpleNamespace(name="scan"),
        _whole_scan_aggregate=lambda *, dim, method: calls.append((dim, method)) or agg,
        ui=SimpleNamespace(
            plotUnit=SimpleNamespace(currentIndex=lambda: 0, currentText=lambda: "Q (Å⁻¹)"),
            slice=SimpleNamespace(isChecked=lambda: False),
        ),
    )
    adapter = PublicationDisplayAdapter(store=None, widget=widget)
    state = _fake_state(overall=True, selected_ids=(0,), render_ids=(), method="Average")
    payload = adapter.integration_plot_payload(state)
    assert payload is not None
    assert calls == [("1d", "average")]
    assert len(payload.traces) == 1
    np.testing.assert_allclose(payload.traces[0].x, q)
    np.testing.assert_allclose(payload.traces[0].y, 15.5)


def test_plot_payload_blocks_nonprimary_gi_instead_of_falling_back():
    from xdart.gui.tabs.static_scan.display_publication import (
        PublicationDisplayAdapter)

    widget = SimpleNamespace(
        scan=SimpleNamespace(name="scan"),
        _aggregate_display_is_primary=lambda _scan, _dim: False,
        _whole_scan_aggregate=lambda *, dim, method: pytest.fail(
            "non-primary GI aggregate must not read a partial stack"),
    )
    adapter = PublicationDisplayAdapter(store=None, widget=widget)
    state = _fake_state(overall=True, selected_ids=(0,), render_ids=(), method="Average")

    payload = adapter.integration_plot_payload(state)

    assert payload is not None
    assert payload.traces == ()


def test_overlay_waterfall_payload_accumulates_in_payload_across_renders():
    # Flip stage 2/3: the Overlay/Waterfall accumulator is carried IN the payload
    # (plot_history) and accumulated across renders -- NOT rebuilt from the store
    # each render (which the cap would truncate).  Verifies: cross-render append;
    # a render with no resident frames PRESERVES the accumulator; a display
    # generation bump (selection growth) does NOT reset (the cap-truncation fix);
    # an incompatible grid DOES reset.
    from xdart.gui.tabs.static_scan.display_publication import (
        PublicationDisplayAdapter)
    from xdart.modules.frame_publication import (
        PublicationStore, publication_from_live_frame)
    from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D

    def _ir1d(v, nq=5):
        return IntegrationResult1D(
            radial=np.linspace(0.5, 5.0, nq, dtype=np.float32),
            intensity=np.full(nq, float(v), dtype=np.float32),
            sigma=np.ones(nq, dtype=np.float32), unit="q_A^-1")

    store = PublicationStore(max_heavy_items=None)
    for i in range(3):
        frame = SimpleNamespace(
            idx=i, int_1d=_ir1d(i + 1), int_2d=None, map_raw=None, mask=None,
            gi=False, gi_2d={}, thumbnail=None, bg_raw=0, scan_info={},
            source_file=f"f{i}.tif", source_frame_idx=i)
        store.upsert(publication_from_live_frame(frame))

    widget = SimpleNamespace(
        scan=SimpleNamespace(name="scan", gi=False, bai_1d_args={}, bai_2d_args={}),
        normChannel=None,
        _waterfall_history=None,
        ui=SimpleNamespace(
            plotUnit=SimpleNamespace(currentText=lambda: "Q (Å⁻¹)",
                                     currentIndex=lambda: 0),
            slice=SimpleNamespace(isChecked=lambda: False, isEnabled=lambda: False)),
    )

    def render(render_ids, generation=1):
        adapter = PublicationDisplayAdapter(store, widget=widget, labels=render_ids)
        st = SimpleNamespace(render_ids=tuple(render_ids),
                             selected_ids=tuple(render_ids),
                             generation=generation, method="Overlay", overall=True)
        p = adapter._overlay_waterfall_payload(st)
        if p is not None:                       # the renderer stores it back
            widget._waterfall_history = p.plot_history
        return p

    q = lambda scan, idx: (scan, idx)

    p1 = render([0, 1])
    assert p1 is not None and p1.plot_history.ids == (q("scan", 0), q("scan", 1))
    assert len(p1.traces) == 2
    p2 = render([2])
    assert p2.plot_history.ids == (q("scan", 0), q("scan", 1), q("scan", 2))
    p3 = render([])                               # no resident frames -> preserve
    assert p3 is not None and p3.plot_history.ids == (
        q("scan", 0), q("scan", 1), q("scan", 2))
    # A display-generation bump (selection growth) must NOT reset -- the accumulator
    # is keyed on the grid/source identity, not the generation.
    p4 = render([0], generation=2)
    assert p4.plot_history.ids == (q("scan", 0), q("scan", 1), q("scan", 2))
    # A compatible scan change APPENDS.  Frame 0 in scan2 is distinct from frame 0
    # in scan because ids are scan-qualified.
    widget.scan = SimpleNamespace(name="scan2", gi=False, bai_1d_args={}, bai_2d_args={})
    p5 = render([0])
    assert p5.plot_history.ids == (
        q("scan", 0), q("scan", 1), q("scan", 2), q("scan2", 0))

    frame = SimpleNamespace(
        idx=4, int_1d=_ir1d(4, nq=7), int_2d=None, map_raw=None, mask=None,
        gi=False, gi_2d={}, thumbnail=None, bg_raw=0, scan_info={},
        source_file="f4.tif", source_frame_idx=4)
    store.upsert(publication_from_live_frame(frame))
    widget.scan = SimpleNamespace(name="scan3", gi=False, bai_1d_args={}, bai_2d_args={})
    p6 = render([4])
    assert p6.plot_history.ids == (q("scan3", 4),)


def test_plot_payload_routes_overlay_waterfall_through_accumulator():
    # Flip stage 3: plot_payload now RETURNS the WaterfallHistory-carrying payload
    # for Overlay/Waterfall in the integration modes (was None -> legacy update_plot).
    # Single/Sum/Average keep the integration payload with NO accumulator.
    from xdart.gui.tabs.static_scan.display_publication import (
        PublicationDisplayAdapter)
    from xdart.gui.tabs.static_scan.display_logic import Mode
    from xdart.modules.frame_publication import (
        PublicationStore, publication_from_live_frame)
    from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D

    def _ir1d(v, nq=5):
        return IntegrationResult1D(
            radial=np.linspace(0.5, 5.0, nq, dtype=np.float32),
            intensity=np.full(nq, float(v), dtype=np.float32),
            sigma=np.ones(nq, dtype=np.float32), unit="q_A^-1")

    store = PublicationStore(max_heavy_items=None)
    for i in range(3):
        frame = SimpleNamespace(
            idx=i, int_1d=_ir1d(i + 1), int_2d=None, map_raw=None, mask=None,
            gi=False, gi_2d={}, thumbnail=None, bg_raw=0, scan_info={},
            source_file=f"f{i}.tif", source_frame_idx=i)
        store.upsert(publication_from_live_frame(frame))

    widget = SimpleNamespace(
        scan=SimpleNamespace(name="scan", gi=False, bai_1d_args={}, bai_2d_args={}),
        normChannel=None, _waterfall_history=None,
        ui=SimpleNamespace(
            plotUnit=SimpleNamespace(currentText=lambda: "Q (Å⁻¹)",
                                     currentIndex=lambda: 0),
            slice=SimpleNamespace(isChecked=lambda: False, isEnabled=lambda: False)),
    )

    def state(method, render_ids, generation=1):
        return SimpleNamespace(
            mode=Mode.INT_1D, method=method, generation=generation,
            overall=True, selected_ids=tuple(render_ids),
            render_ids=tuple(render_ids),
            panel=lambda role: SimpleNamespace(has_data=True, source=None))

    adapter = PublicationDisplayAdapter(store, widget=widget, labels=(0, 1, 2))

    # Overlay routes to the accumulator and accumulates across renders.
    p1 = adapter.plot_payload(state("Overlay", [0, 1]))
    assert p1 is not None and p1.plot_history is not None
    assert p1.plot_history.ids == (("scan", 0), ("scan", 1))
    assert p1.overlaid_ids == (("scan", 0), ("scan", 1))
    widget._waterfall_history = p1.plot_history          # renderer stores it back
    p2 = adapter.plot_payload(state("Waterfall", [2]))
    assert p2.plot_history.ids == (("scan", 0), ("scan", 1), ("scan", 2))

    # Single/Sum/Average keep the integration payload, NO accumulator carried.
    widget._waterfall_history = None
    ps = adapter.plot_payload(state("Single", [0]))
    assert ps is not None
    assert ps.plot_history is None and ps.overlaid_ids is None


def test_ov7c_live_current_uses_next_pin_slot_and_pin_freezes_in_place():
    from xdart.gui.tabs.static_scan.display_frame_widget import displayFrameWidget
    from xdart.gui.tabs.static_scan.display_overlay_utils import (
        LIVE_SLICE_PROJECTION_ID,
        overlay_identity_for_widget,
    )
    from xdart.gui.tabs.static_scan.display_publication import (
        PublicationDisplayAdapter)
    from xdart.gui.tabs.static_scan.display_logic import WaterfallHistory
    from xdart.modules.frame_publication import (
        PublicationStore, publication_from_frame_view)
    from xrd_tools.core import FrameView
    from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D

    class _Control:
        def __init__(self, *, text="", value=None, checked=False, enabled=True):
            self._text = text
            self._value = value
            self._checked = checked
            self._enabled = enabled

        def currentText(self):
            return self._text

        def currentIndex(self):
            return 0

        def text(self):
            return self._text

        def value(self):
            return self._value["value"] if isinstance(self._value, dict) else self._value

        def isChecked(self):
            return self._checked

        def isEnabled(self):
            return self._enabled

    q = np.linspace(0.5, 3.0, 6, dtype=np.float32)
    chi = np.asarray([-20.0, -10.0, 0.0, 10.0, 20.0], dtype=np.float32)
    intensity = np.arange(q.size * chi.size, dtype=np.float32).reshape(q.size, chi.size)
    view = FrameView.from_results(
        label=1,
        result_1d=IntegrationResult1D(
            radial=q, intensity=np.ones(q.shape, dtype=np.float32),
            sigma=None, unit="q_A^-1"),
        result_2d=IntegrationResult2D(
            radial=q, azimuthal=chi, intensity=intensity,
            unit="q_A^-1", azimuthal_unit="chi_deg"),
    )
    store = PublicationStore(max_heavy_items=None)
    store.upsert(publication_from_frame_view(view))

    center = {"value": -10.0}
    axis_info = {"source": "2d", "axis": "radial", "slice_axis": "chi"}
    repaint_reasons = []
    widget = SimpleNamespace(
        scan=SimpleNamespace(name="scan", gi=False, bai_1d_args={}, bai_2d_args={}),
        normChannel=None,
        _waterfall_history=None,
        overlaid_idxs=[],
        idxs_1d=[1],
        frame_ids=["1"],
        _plot_axis_info=(axis_info,),
        _slice_2d_data_ready=lambda: True,
        request_current_selection_repaint=lambda **kw: repaint_reasons.append(kw),
        ui=SimpleNamespace(
            plotMethod=_Control(text="Overlay"),
            plotUnit=_Control(text="Q (Å⁻¹)"),
            imageUnit=_Control(text="χ-Q"),
            slice=_Control(text="χ (c/w)", checked=True, enabled=True),
            slice_center=_Control(value=center),
            slice_width=_Control(value={"value": 2.0}),
        ),
    )
    widget._slice_pin_selection = MethodType(
        displayFrameWidget._slice_pin_selection, widget)
    widget._slice_pin_trace_name = MethodType(
        displayFrameWidget._slice_pin_trace_name, widget)
    widget.pin_current_slice_cut = MethodType(
        displayFrameWidget.pin_current_slice_cut, widget)
    widget._pinned_slice_cut_recipes = lambda: tuple(
        (getattr(widget, "_pinned_slice_cuts", None) or {}).values())

    def render():
        state = SimpleNamespace(
            render_ids=(1,), selected_ids=(1,), generation=1,
            method="Overlay", overall=False)
        payload = PublicationDisplayAdapter(
            store, widget=widget, labels=(1,))._overlay_waterfall_payload(state)
        assert payload is not None and payload.plot_history is not None
        widget._waterfall_history = payload.plot_history
        widget.overlaid_idxs = list(payload.overlaid_ids)
        return payload

    def pin_at(value):
        center["value"] = value
        assert widget.pin_current_slice_cut() is True

    def live_positions(history):
        return [
            pos for pos, row_id in enumerate(history.ids)
            if isinstance(row_id, tuple)
            and len(row_id) >= 3
            and row_id[2][0] == LIVE_SLICE_PROJECTION_ID
        ]

    def pinned_position(history, value):
        for pos, row_id in enumerate(history.ids):
            if not (isinstance(row_id, tuple) and len(row_id) >= 3):
                continue
            projection = row_id[2]
            if (isinstance(projection, tuple)
                    and projection[0] == "chi"
                    and projection[1] == pytest.approx(value)):
                return pos
        raise AssertionError(f"missing pinned center {value}")

    pin_at(-10.0)
    pin_at(0.0)
    assert repaint_reasons

    pinned_recipes = tuple(widget._pinned_slice_cuts.values())
    pinned_ids = tuple(recipe["row_id"] for recipe in pinned_recipes)
    pinned_names = tuple(recipe["name"] for recipe in pinned_recipes)
    _grid_key, live_row_id = overlay_identity_for_widget(
        widget, 1, axis_info=axis_info, projection_id=None, live_slice=True)
    # Seed the exact residual shape OV-7c closes: a transient live current row
    # stranded below the pins.  The renderer's y offset is row_index*offset_value,
    # so the live row must move from slot 0 to slot 2 before it is visible.
    widget._waterfall_history = WaterfallHistory(
        reset_key=pinned_recipes[0]["reset_key"],
        unit="q_A^-1",
        label="Q",
        x=q,
        rows=np.vstack([
            np.full(q.shape, 90.0),
            np.full(q.shape, 10.0),
            np.full(q.shape, 20.0),
        ]),
        ids=(live_row_id, *pinned_ids),
        names=("scan_1 · Q@χ=old · current", *pinned_names),
        metadata=({}, {}, {}),
    )
    widget.overlaid_idxs = list(widget._waterfall_history.ids)

    offset_value = 7.5
    center["value"] = 10.0
    payload = render()
    assert live_positions(payload.plot_history) == [2]
    assert live_positions(payload.plot_history)[0] * offset_value == pytest.approx(
        2 * offset_value)

    before_pin_slot = live_positions(payload.plot_history)[0]
    pin_at(10.0)
    payload = render()
    assert live_positions(payload.plot_history) == []
    assert pinned_position(payload.plot_history, 10.0) == before_pin_slot
    assert "current" not in payload.plot_history.names[before_pin_slot]

    center["value"] = 20.0
    payload = render()
    assert live_positions(payload.plot_history) == [3]


def test_overlay_selection_of_evicted_frame_preserves_then_appends():
    from xdart.gui.tabs.static_scan.display_controllers import ScanDisplayController
    from xdart.gui.tabs.static_scan.display_logic import Mode, WaterfallHistory
    from xdart.gui.tabs.static_scan.display_publication import (
        PublicationDisplayAdapter)
    from xdart.modules.frame_publication import (
        PublicationStore, publication_from_live_frame)
    from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D

    x = np.linspace(0.5, 5.0, 8, dtype=np.float32)

    def _frame(i):
        return SimpleNamespace(
            idx=i,
            int_1d=IntegrationResult1D(
                radial=x, intensity=np.full(x.shape, float(i), dtype=np.float32),
                sigma=np.ones_like(x), unit="q_A^-1"),
            int_2d=IntegrationResult2D(
                radial=np.linspace(0.5, 5.0, 4, dtype=np.float32),
                azimuthal=np.linspace(-90.0, 90.0, 3, dtype=np.float32),
                intensity=np.full((4, 3), float(i), dtype=np.float32),
                unit="q_A^-1", azimuthal_unit="chi_deg"),
            map_raw=None, mask=None, gi=False, gi_2d={},
            thumbnail=None, bg_raw=0, scan_info={}, source_file=f"f{i}.tif",
            source_frame_idx=i,
        )

    store = PublicationStore(max_heavy_items=64)
    for i in range(100):
        store.upsert(publication_from_live_frame(_frame(i)))
    assert not store.get(5).view.has_1d
    assert store.get(99).view.has_1d

    def _history():
        frame_ids = tuple(range(36, 100))
        ids = tuple(("scan", i) for i in frame_ids)
        return WaterfallHistory(
            reset_key=("radial", len(x), False),
            unit="Å⁻¹",
            label="Q",
            x=x,
            rows=np.vstack([np.full(x.shape, float(i)) for i in frame_ids]),
            ids=ids,
            names=tuple(f"scan_{i}" for i in frame_ids),
        )

    queued = []
    widget = SimpleNamespace(
        publication_store=store,
        viewer_mode=None,
        data_lock=RLock(),
        viewer_rows_1d={},
        viewer_rows_2d={},
        frame_ids=["5"],
        overlaid_idxs=list(_history().ids),
        _waterfall_history=_history(),
        display_generation=1,
        normChannel=None,
        scan=SimpleNamespace(
            name="scan", data_file="scan.nxs", gi=False,
            bai_1d_args={}, bai_2d_args={},
            scan_lock=RLock(),
            frames=SimpleNamespace(index=list(range(100))),
        ),
        ui=SimpleNamespace(
            plotMethod=SimpleNamespace(currentText=lambda: "Overlay"),
            plotUnit=SimpleNamespace(currentText=lambda: "Q (Å⁻¹)",
                                     currentIndex=lambda: 0),
            slice=SimpleNamespace(isChecked=lambda: False, isEnabled=lambda: False),
        ),
        _request_missing_publication=lambda label: queued.append(int(label)),
    )

    def render():
        state = ScanDisplayController().compute_state(widget, Mode.INT_1D)
        labels = tuple(dict.fromkeys((*state.selected_ids, *state.render_ids)))
        adapter = PublicationDisplayAdapter(store, widget=widget, labels=labels)
        payload = adapter.plot_payload(state)
        if payload is not None and payload.plot_history is not None:
            widget._waterfall_history = payload.plot_history
            widget.overlaid_idxs = list(payload.overlaid_ids)
        return state, payload

    state, payload = render()
    assert tuple(state.render_ids) == ()
    assert queued == [5]
    assert payload is not None
    assert payload.plot_history.ids == _history().ids

    store.upsert(publication_from_live_frame(_frame(5)))
    state, payload = render()
    assert tuple(state.render_ids) == (5,)
    assert payload.plot_history.ids == _history().ids + (("scan", 5),)

    queued.clear()
    widget._waterfall_history = _history()
    widget.overlaid_idxs = list(widget._waterfall_history.ids)
    widget.frame_ids = ["6", "99"]
    state, payload = render()
    assert tuple(state.render_ids) == (99,)
    assert queued == [6]
    assert payload.plot_history.ids == _history().ids

    store.upsert(publication_from_live_frame(_frame(6)))
    state, payload = render()
    assert tuple(state.render_ids) == (6, 99)
    assert payload.plot_history.count == 65
    assert set(payload.plot_history.ids) >= {("scan", 6), ("scan", 99)}


def test_overlay_many_frame_selection_converges_after_out_of_order_hydration():
    from xdart.gui.tabs.static_scan.display_controllers import ScanDisplayController
    from xdart.gui.tabs.static_scan.display_logic import Mode
    from xdart.gui.tabs.static_scan.display_publication import (
        PublicationDisplayAdapter)
    from xdart.modules.frame_publication import (
        PublicationStore, publication_from_live_frame)
    from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D

    x = np.linspace(0.5, 5.0, 8, dtype=np.float32)

    def _frame(i):
        return SimpleNamespace(
            idx=i,
            int_1d=IntegrationResult1D(
                radial=x, intensity=np.full(x.shape, float(i), dtype=np.float32),
                sigma=np.ones_like(x), unit="q_A^-1"),
            int_2d=IntegrationResult2D(
                radial=np.linspace(0.5, 5.0, 4, dtype=np.float32),
                azimuthal=np.linspace(-90.0, 90.0, 3, dtype=np.float32),
                intensity=np.full((4, 3), float(i), dtype=np.float32),
                unit="q_A^-1", azimuthal_unit="chi_deg"),
            map_raw=None, mask=None, gi=False, gi_2d={},
            thumbnail=None, bg_raw=0, scan_info={}, source_file=f"f{i}.tif",
            source_frame_idx=i,
        )

    store = PublicationStore(max_heavy_items=64)
    for i in range(100):
        store.upsert(publication_from_live_frame(_frame(i)))
    evicted_at_start = {
        i for i in range(100) if not store.get(i).view.has_1d
    }
    assert evicted_at_start and 99 not in evicted_at_start

    queued = set()
    widget = SimpleNamespace(
        publication_store=store,
        viewer_mode=None,
        data_lock=RLock(),
        viewer_rows_1d={},
        viewer_rows_2d={},
        frame_ids=[str(i) for i in range(100)],
        overlaid_idxs=[],
        _waterfall_history=None,
        display_generation=1,
        normChannel=None,
        scan=SimpleNamespace(
            name="scan", data_file="scan.nxs", gi=False,
            bai_1d_args={}, bai_2d_args={},
            scan_lock=RLock(),
            frames=SimpleNamespace(index=list(range(100))),
        ),
        ui=SimpleNamespace(
            plotMethod=SimpleNamespace(currentText=lambda: "Overlay"),
            plotUnit=SimpleNamespace(currentText=lambda: "Q (Å⁻¹)",
                                     currentIndex=lambda: 0),
            slice=SimpleNamespace(isChecked=lambda: False, isEnabled=lambda: False),
        ),
        _request_missing_publication=lambda label: queued.add(int(label)),
    )

    def render():
        state = ScanDisplayController().compute_state(widget, Mode.INT_1D)
        labels = tuple(dict.fromkeys((*state.selected_ids, *state.render_ids)))
        adapter = PublicationDisplayAdapter(store, widget=widget, labels=labels)
        payload = adapter.plot_payload(state)
        if payload is not None and payload.plot_history is not None:
            widget._waterfall_history = payload.plot_history
            widget.overlaid_idxs = list(payload.overlaid_ids)
        return state, payload

    state, payload = render()
    assert payload is not None and payload.plot_history is not None
    assert payload.plot_history.count == 100 - len(evicted_at_start)
    assert queued == evicted_at_start

    completion_order = sorted(evicted_at_start)[::2] + sorted(evicted_at_start)[1::2]
    for label in completion_order:
        store.upsert(publication_from_live_frame(_frame(label)))
        state, payload = render()
        assert payload is not None and payload.plot_history is not None

    assert payload.plot_history.count == 100
    assert set(payload.plot_history.ids) == {("scan", i) for i in range(100)}


def test_overlay_selection_evicted_hydration_never_decreases_history():
    from xdart.gui.tabs.static_scan.display_controllers import ScanDisplayController
    from xdart.gui.tabs.static_scan.display_logic import Mode, WaterfallHistory
    from xdart.gui.tabs.static_scan.display_frame_widget import displayFrameWidget
    from xdart.gui.tabs.static_scan.display_publication import (
        PublicationDisplayAdapter)
    from xdart.modules.frame_publication import (
        PublicationStore, publication_from_live_frame)
    from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D

    x = np.linspace(0.5, 5.0, 8, dtype=np.float32)

    def _frame(i):
        return SimpleNamespace(
            idx=i,
            int_1d=IntegrationResult1D(
                radial=x, intensity=np.full(x.shape, float(i), dtype=np.float32),
                sigma=np.ones_like(x), unit="q_A^-1"),
            int_2d=IntegrationResult2D(
                radial=np.linspace(0.5, 5.0, 4, dtype=np.float32),
                azimuthal=np.linspace(-90.0, 90.0, 3, dtype=np.float32),
                intensity=np.full((4, 3), float(i), dtype=np.float32),
                unit="q_A^-1", azimuthal_unit="chi_deg"),
            map_raw=None, mask=None, gi=False, gi_2d={},
            thumbnail=None, bg_raw=0, scan_info={}, source_file=f"f{i}.tif",
            source_frame_idx=i,
        )

    store = PublicationStore(max_heavy_items=4)
    for i in range(320, 336):
        store.upsert(publication_from_live_frame(_frame(i)))
    assert not store.get(331).view.has_1d

    initial_frames = tuple(range(320, 331))
    initial_ids = tuple(("scan", i) for i in initial_frames)
    initial_history = WaterfallHistory(
        reset_key=("radial", len(x), False),
        unit="Å⁻¹",
        label="Q",
        x=x,
        rows=np.vstack([np.full(x.shape, float(i)) for i in initial_frames]),
        ids=initial_ids,
        names=tuple(f"scan_{i}" for i in initial_frames),
    )
    queued = []
    plot_unit = {"text": "Q (Å⁻¹)", "index": 0}
    widget = SimpleNamespace(
        publication_store=store,
        viewer_mode=None,
        data_lock=RLock(),
        viewer_rows_1d={},
        viewer_rows_2d={},
        frame_ids=["331"],
        overlaid_idxs=list(initial_history.ids),
        _waterfall_history=initial_history,
        plot_data=[x, initial_history.rows],
        plot_data_range=[[float(x[0]), float(x[-1])], [320.0, 330.0]],
        frame_names=list(initial_history.names),
        display_generation=1,
        normChannel=None,
        scan=SimpleNamespace(
            name="scan", data_file="scan.nxs", gi=False,
            bai_1d_args={}, bai_2d_args={},
            scan_lock=RLock(),
            frames=SimpleNamespace(index=list(range(320, 336))),
        ),
        ui=SimpleNamespace(
            plotMethod=SimpleNamespace(currentText=lambda: "Overlay"),
            plotUnit=SimpleNamespace(
                currentText=lambda: plot_unit["text"],
                currentIndex=lambda: plot_unit["index"],
            ),
            slice=SimpleNamespace(isChecked=lambda: False, isEnabled=lambda: False),
        ),
        _request_frame_hydration=lambda label, *, purpose="full":
            queued.append((int(label), purpose)),
    )
    counts = [initial_history.count]

    def render_and_guard():
        state = ScanDisplayController().compute_state(widget, Mode.INT_1D)
        labels = tuple(dict.fromkeys((*state.selected_ids, *state.render_ids)))
        adapter = PublicationDisplayAdapter(store, widget=widget, labels=labels)
        payload = adapter.plot_payload(state)
        assert payload is not None and payload.plot_history is not None
        assert payload.plot_history.count >= counts[-1]
        counts.append(payload.plot_history.count)
        widget._waterfall_history = payload.plot_history
        widget.overlaid_idxs = list(payload.overlaid_ids)
        return state, payload

    state, payload = render_and_guard()
    assert tuple(state.render_ids) == ()
    assert queued == [(331, "1d")]
    assert payload.plot_history.ids == initial_ids

    widget.frame_ids = []
    state, payload = render_and_guard()
    assert tuple(state.render_ids) == ()
    assert payload.plot_history.ids == initial_ids

    plot_unit.update({"text": "2θ (°)", "index": 1})
    state, payload = render_and_guard()
    assert tuple(state.render_ids) == ()
    assert payload.plot_history.ids == initial_ids

    store.upsert(publication_from_live_frame(_frame(331)))
    state, payload = render_and_guard()
    assert tuple(state.render_ids) == ()
    assert payload.plot_history.ids == initial_ids

    widget.frame_ids = ["331"]
    state, payload = render_and_guard()

    assert tuple(state.render_ids) == (331,)
    assert counts == [11, 11, 11, 11, 11, 12]
    assert payload.plot_history.ids == initial_ids + (("scan", 331),)

    widget.clear_overlay = MethodType(displayFrameWidget.clear_overlay, widget)
    widget.clear_overlay()
    assert widget._waterfall_history is None
    assert widget.overlaid_idxs == []


def test_waterfall_history_payload_decimates_display_rows_only():
    from xdart.gui.tabs.static_scan.display_logic import WaterfallHistory
    from xdart.gui.tabs.static_scan.display_publication import (
        MAX_WATERFALL_PAYLOAD_ROWS,
        PublicationDisplayAdapter,
    )

    x = np.linspace(0.5, 5.0, 8)
    ids = tuple(range(700))
    history = WaterfallHistory(
        reset_key=("radial", len(x), False),
        unit="Å⁻¹",
        label="Q",
        x=x,
        rows=np.vstack([np.full(x.shape, float(i)) for i in ids]),
        ids=ids,
        names=tuple(f"scan_{i}" for i in ids),
    )
    payload = PublicationDisplayAdapter(store=None)._history_to_payload(history)

    assert payload.plot_history is history
    assert payload.overlaid_ids == ids
    assert len(payload.traces) <= MAX_WATERFALL_PAYLOAD_ROWS
    assert payload.display_ids == ids[::3]


def test_cake_image_blanks_non_overall_eviction():
    # A non-Overall (explicit subset) selection with an evicted frame must still
    # blank — the aggregate is whole-scan only, not an arbitrary subset.
    from xdart.gui.tabs.static_scan.display_publication import (
        PublicationDisplayAdapter)
    widget = SimpleNamespace(
        bkg_2d=0,
        _whole_scan_aggregate=lambda *, dim, method: pytest.fail(
            "must not aggregate a non-Overall subset"),
    )
    adapter = PublicationDisplayAdapter(store=None, widget=widget)
    state = _fake_state(overall=False, selected_ids=(0, 1), render_ids=())
    assert adapter.cake_image(state) is None


# ── off-GUI-thread worker ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def qapp():
    from pyqtgraph.Qt import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def test_aggregation_worker_computes_off_thread(qapp, tmp_path):
    import threading
    from pyqtgraph import Qt
    from xdart.gui.tabs.static_scan.aggregation_worker import AggregationWorker
    _DIRECT = Qt.QtCore.Qt.ConnectionType.DirectConnection
    scan, _, _ = _split_scan_2d(tmp_path, n=12)
    caller = threading.get_ident()
    got, done = [], threading.Event()
    worker = AggregationWorker()
    # DirectConnection -> the slot runs ON the worker thread, so no event loop.
    worker.sigAggregated.connect(
        lambda key, gen, res: (got.append((key, gen, res, threading.get_ident())),
                               done.set()), _DIRECT)
    worker.start()
    try:
        worker.request(("2d", "average", None), 7, scan, "2d", "average", None)
        assert done.wait(10.0), "worker never emitted sigAggregated"
    finally:
        worker.stop()
    key, gen, res, ran_on = got[0]
    assert gen == 7 and res is not None
    assert ran_on != caller                       # computed OFF the caller thread
    np.testing.assert_allclose(res.intensity, np.mean(np.arange(1, 13)))   # 6.5


# ── thumbnail gap-mask re-application (end-of-scan last-frame regression) ──────
# Detector module gaps are stored as 0-valued pixels and only become NaN (white)
# via the detector mask.  The full-res raw path applies it directly; the thumbnail
# path normally relies on the mask being BAKED into the preview at creation.  A
# frame whose thumbnail lacks the bake — notably the last frame persisted at
# end-of-scan, in Overlay/thumbnail render mode — showed gaps as 0 (dark) instead
# of NaN.  _nan_thumbnail_gaps re-applies the gap mask in thumbnail coordinates.

def _gap_mask_flat(H, W, lo, hi):
    """Flat indices of full-width detector rows [lo, hi] in an (H, W) detector."""
    rows = np.arange(H * W) // W
    return np.flatnonzero((rows >= lo) & (rows <= hi))


def _nan_thumb_host(full_shape, gap_flat):
    from types import MethodType
    from xdart.gui.tabs.static_scan.display_frame_widget import displayFrameWidget
    host = SimpleNamespace(
        _raw_full_shape=full_shape,
        scan=SimpleNamespace(global_mask=gap_flat),
    )
    host._nan_thumbnail_gaps = MethodType(
        displayFrameWidget._nan_thumbnail_gaps, host)
    return host


def test_nan_thumbnail_gaps_masks_downsampled_gap_rows():
    # full-res rows 48-51 of a 100x100 detector -> thumbnail rows 24-25 of 50x50
    host = _nan_thumb_host((100, 100), _gap_mask_flat(100, 100, 48, 51))
    data = np.ones((50, 50), dtype=float)          # unbaked thumbnail: no NaN
    host._nan_thumbnail_gaps(data)
    assert np.isnan(data[24:26, :]).all()          # gap rows masked to NaN
    assert not np.isnan(data[:24, :]).any()        # everything else untouched
    assert not np.isnan(data[26:, :]).any()


def test_nan_thumbnail_gaps_noop_without_cached_full_shape():
    # Without a known full-res shape the flat indices can't be mapped — must
    # leave the thumbnail untouched rather than corrupt unrelated pixels.
    host = _nan_thumb_host(None, _gap_mask_flat(100, 100, 48, 51))
    data = np.ones((50, 50), dtype=float)
    host._nan_thumbnail_gaps(data)
    assert not np.isnan(data).any()


def test_nan_thumbnail_gaps_noop_without_gap_mask():
    host = _nan_thumb_host((100, 100), None)
    data = np.ones((50, 50), dtype=float)
    host._nan_thumbnail_gaps(data)
    assert not np.isnan(data).any()


def test_raw_image_payload_bakes_thumbnail_gaps():
    # The raw panel renders the current frame as a thumbnail (Single/Overlay/
    # Waterfall).  Detector module gaps (0-valued, not sentinels) must render as
    # NaN even when the thumbnail was generated without them -- raw_image bakes
    # the gap mask into the downsampled image + carries the metadata, so the
    # payload path masks gaps identically to the legacy update_image path (the
    # structural fix for the Overlay end-of-scan gap bug, ready for the live-
    # gated panel flip).
    from xdart.gui.tabs.static_scan.display_publication import (
        PublicationDisplayAdapter)
    from xdart.gui.tabs.static_scan.display_logic import RawSource
    rows = np.arange(100 * 100) // 100
    gap_flat = np.flatnonzero((rows >= 48) & (rows <= 51))   # full-res gap band
    thumb = np.ones((50, 50), dtype=float)                   # unbaked: no NaN gaps
    pub = SimpleNamespace(
        view=SimpleNamespace(thumbnail=thumb),
        raw_ref=SimpleNamespace(mask=None, bg_raw=0, thumbnail=thumb),
        metadata_raw={},
    )
    store = SimpleNamespace(snapshot=lambda: {0: pub})
    widget = SimpleNamespace(
        scan=SimpleNamespace(global_mask=gap_flat, mask_sentinel=True),
        _raw_full_shape=(100, 100),
        bkg_map_raw=0,
    )
    adapter = PublicationDisplayAdapter(store, widget=widget)
    state = SimpleNamespace(
        overall=False, method="Single", selected_ids=(0,), render_ids=(0,),
        panel=lambda role: SimpleNamespace(has_data=True, source=RawSource.THUMBNAIL),
    )
    payload = adapter.raw_image(state)
    assert payload is not None
    assert payload.raw_full_shape == (100, 100)
    assert payload.gap_mask_indices is not None
    # gaps baked to NaN at the mapped thumbnail rows (24-25; the [::-1,:] flip
    # swaps rows 24<->25 so the masked band stays at 24-25).
    assert np.isnan(payload.image[24:26, :]).all()
    assert np.isfinite(payload.image[:24, :]).all()
    assert np.isfinite(payload.image[26:, :]).all()


def test_raw_image_uses_scan_detector_shape_without_widget_cache():
    # Cold reload into Overlay: no full-res frame seen this session (widget has
    # NO _raw_full_shape), but the scan carries detector_shape persisted in the
    # .nxs.  raw_image must mask the gaps from scan.detector_shape -- the codex-P2
    # fix for the cold-reload dark-gap edge.
    from xdart.gui.tabs.static_scan.display_publication import (
        PublicationDisplayAdapter)
    from xdart.gui.tabs.static_scan.display_logic import RawSource
    gap_flat = _gap_mask_flat(100, 100, 48, 51)
    thumb = np.ones((50, 50), dtype=float)
    pub = SimpleNamespace(
        view=SimpleNamespace(thumbnail=thumb),
        raw_ref=SimpleNamespace(mask=None, bg_raw=0, thumbnail=thumb),
        metadata_raw={},
    )
    store = SimpleNamespace(snapshot=lambda: {0: pub})
    widget = SimpleNamespace(            # NOTE: no _raw_full_shape attr
        scan=SimpleNamespace(global_mask=gap_flat, mask_sentinel=True,
                             detector_shape=(100, 100)),
        bkg_map_raw=0,
    )
    adapter = PublicationDisplayAdapter(store, widget=widget)
    state = SimpleNamespace(
        overall=False, method="Single", selected_ids=(0,), render_ids=(0,),
        panel=lambda role: SimpleNamespace(has_data=True, source=RawSource.THUMBNAIL),
    )
    payload = adapter.raw_image(state)
    assert payload is not None and payload.raw_full_shape == (100, 100)
    assert np.isnan(payload.image[24:26, :]).all()


def test_raw_image_thumbnail_axes_span_true_detector_extent():
    # Universal raw-display policy: a downsampled thumbnail is rect-scaled to the
    # TRUE detector dimensions, so its Pixels axes read 0..(full-1), NOT the
    # thumbnail's own 0..49 (the wrong-dimensions bug Vivek caught in Overlay).
    from xdart.gui.tabs.static_scan.display_publication import (
        PublicationDisplayAdapter)
    from xdart.gui.tabs.static_scan.display_logic import RawSource
    thumb = np.ones((50, 50), dtype=float)
    pub = SimpleNamespace(
        view=SimpleNamespace(thumbnail=thumb),
        raw_ref=SimpleNamespace(mask=None, bg_raw=0, thumbnail=thumb),
        metadata_raw={},
    )
    store = SimpleNamespace(snapshot=lambda: {0: pub})
    widget = SimpleNamespace(
        scan=SimpleNamespace(global_mask=None, mask_sentinel=True,
                             detector_shape=(100, 120)),   # rows=100, cols=120
        bkg_map_raw=0,
    )
    adapter = PublicationDisplayAdapter(store, widget=widget)
    state = SimpleNamespace(
        overall=False, method="Single", selected_ids=(0,), render_ids=(0,),
        panel=lambda role: SimpleNamespace(has_data=True, source=RawSource.THUMBNAIL),
    )
    payload = adapter.raw_image(state)
    assert payload is not None
    # axis_x -> columns (true 120), axis_y -> rows (true 100); the 50x50 thumbnail's
    # axes span the FULL detector extent (still 50 pixel samples each).
    assert payload.axis_x.values[0] == 0.0 and payload.axis_x.values[-1] == 119.0
    assert payload.axis_y.values[0] == 0.0 and payload.axis_y.values[-1] == 99.0
    assert len(payload.axis_x.values) == 50 and len(payload.axis_y.values) == 50


def test_nan_thumbnail_gaps_prefers_scan_detector_shape():
    # The widget helper sources the full-res shape from scan.detector_shape when
    # the live widget cache (_raw_full_shape) is absent.
    from types import MethodType
    from xdart.gui.tabs.static_scan.display_frame_widget import displayFrameWidget
    host = SimpleNamespace(          # NOTE: no _raw_full_shape attr
        scan=SimpleNamespace(global_mask=_gap_mask_flat(100, 100, 48, 51),
                             detector_shape=(100, 100)))
    host._nan_thumbnail_gaps = MethodType(
        displayFrameWidget._nan_thumbnail_gaps, host)
    data = np.ones((50, 50), dtype=float)
    host._nan_thumbnail_gaps(data)
    assert np.isnan(data[24:26, :]).all()
    assert not np.isnan(data[:24, :]).any()


def test_get_frames_map_raw_caches_full_res_shape():
    # A resident full-res raw teaches the widget the detector shape so a later
    # thumbnail-only render can map the flat gap mask into thumbnail coordinates.
    from types import MethodType
    from xdart.gui.tabs.static_scan.display_data import DisplayDataMixin
    mr = np.ones((100, 100), dtype=np.float32)
    host = SimpleNamespace(
        idxs_2d=[0],
        scan=SimpleNamespace(mask_sentinel=True),
        _async_hydration_enabled=False,
        _raw_resolve_failed=set(),
        normalize=lambda data, info: data,
    )
    host._sanitize_display_image = staticmethod(
        DisplayDataMixin._sanitize_display_image)
    host._snapshot_data = lambda idxs, allow_blocking_read=None: {
        int(i): (SimpleNamespace(scan_info={}, thumbnail=None),
                 {"map_raw": mr.copy(), "bg_raw": 0}) for i in idxs}
    host.get_frames_map_raw = MethodType(
        DisplayDataMixin.get_frames_map_raw, host)
    assert getattr(host, "_raw_full_shape", None) is None
    host.get_frames_map_raw([0])
    assert host._raw_full_shape == (100, 100)


@pytest.mark.gui
def test_clear_display_state_resets_raw_full_shape(widget):
    # Regression (adversarial review P1): the cached detector shape used to map
    # the gap mask into thumbnail coordinates must NOT survive a scan/file change
    # -- a stale shape from a different-size detector would NaN the wrong
    # thumbnail pixels.  clear_display_state re-arms it alongside the other
    # per-scan display caches.
    df = widget.displayframe
    df._raw_full_shape = (512, 512)
    df.clear_display_state()
    assert df._raw_full_shape is None
