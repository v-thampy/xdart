# -*- coding: utf-8 -*-
"""Wiring of the whole-scan aggregate into the live display (Step 7b A1b-2c).

The user-reported bug: a live Int 2D Overall cake goes BLANK on Stop for a scan
longer than the bounded store (>64 frames) — §2.C correctly refuses to average a
wrong (store-resident-only) subset, but nothing filled that blank.  These tests
lock the fill: the widget computes the whole-scan aggregate (disk ⊕ tail) and the
cake adapter routes the §2.C blank to it instead of returning None.
"""

from __future__ import annotations

from types import SimpleNamespace

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
        w.data_1d.clear()
        w.data_2d.clear()
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
        w.data_1d.clear()
        w.data_2d.clear()
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
    df.ui.setBkg.setText("Set Bkg")

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
