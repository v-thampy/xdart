# -*- coding: utf-8 -*-
"""Greenfield Phase 3 / D2: the displayFrameWidget rehydration source + the
request plumbing that feeds the background FrameHydrationWorker.

The worker thread itself is covered by test_frame_hydration_worker.py; here we
test, headlessly, that (a) _rehydrate_publication turns an evicted disk frame
into a heavy publication, (b) the shared store rehydrates through it, and (c)
the GUI render path only queues a background request when async hydration was
explicitly enabled (off in headless tests -> synchronous reads preserved)."""
import logging
import threading
from collections import deque
from types import SimpleNamespace, MethodType

import numpy as np

from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
from xdart.gui.tabs.static_scan.display_data import DisplayDataMixin
from xdart.gui.tabs.static_scan.display_logic import ConsumerKind
from xdart.gui.tabs.static_scan.display_frame_widget import displayFrameWidget
from xdart.modules.frame_publication import PublicationStore


class _DuckFrame:
    """Minimal LiveFrame-like accepted by publication_from_live_frame."""

    def __init__(self, idx=1):
        self.idx = idx
        self.gi = False
        self.scan_info = {"th": 0.25, "monitor": 100.0}
        self.source_file = "raw_0001.tif"
        self.source_frame_idx = 0
        self.map_raw = np.arange(16, dtype=float).reshape(4, 4)
        self.thumbnail = np.arange(4, dtype=float).reshape(2, 2)
        self.int_1d = IntegrationResult1D(
            radial=np.linspace(0.5, 3.0, 6),
            intensity=np.linspace(10.0, 20.0, 6),
            sigma=np.ones(6), unit="q_A^-1")
        self.int_2d = IntegrationResult2D(
            radial=np.linspace(0.5, 3.0, 4),
            azimuthal=np.linspace(-90.0, 90.0, 3),
            intensity=np.ones((4, 3)), unit="q_A^-1", azimuthal_unit="chi_deg")

    def _get_incident_angle(self):
        return float(self.scan_info["th"])


def _fake_1d_result(frame):
    frame = tuple(int(label) for label in frame)
    return SimpleNamespace(
        q=np.array([0.1, 0.2, 0.3]),
        intensity=np.vstack([
            np.full(3, float(label)) for label in frame
        ]),
        sigma=None,
        q_unit="q_A^-1",
        frames=np.asarray(frame, dtype=np.int64),
    )


def _hydrator_holder(disk_frame, store=None):
    h = SimpleNamespace(publication_store=store or PublicationStore())
    h._hydrate_frame_from_disk = lambda idx, **kw: disk_frame
    h._rehydrate_publication = MethodType(
        DisplayDataMixin._rehydrate_publication, h)
    return h


def test_rehydrate_publication_builds_heavy_from_disk():
    h = _hydrator_holder(_DuckFrame(idx=3))
    pub = h._rehydrate_publication(3)
    assert pub is not None
    assert pub.view.intensity_1d is not None
    assert pub.view.intensity_2d is not None
    assert pub.view.raw is not None           # include_raw=True


def test_rehydrate_returns_none_on_disk_miss():
    h = _hydrator_holder(None)                # disk read found nothing
    assert h._rehydrate_publication(9) is None


def test_store_get_or_hydrate_uses_registered_rehydrator():
    store = PublicationStore()
    h = _hydrator_holder(_DuckFrame(idx=7), store=store)
    store.set_hydrator(h._rehydrate_publication)
    # nothing resident for 7 -> the store pulls it through the rehydrator
    pub = store.get_or_hydrate(7)
    assert pub is not None and pub.view.intensity_2d is not None
    # and now it is resident (a second call needs no hydration)
    assert store.get(7) is not None


def test_store_get_1d_many_uses_batch_1d_rehydrator(monkeypatch):
    import xrd_tools.io as xrd_io

    calls = []

    def fake_get_1d(path, frame):
        calls.append((path, tuple(frame)))
        return _fake_1d_result(frame)

    monkeypatch.setattr(xrd_io, "get_1d", fake_get_1d)
    store = PublicationStore(max_heavy_items=0, max_thumbnail_items=0)
    h = SimpleNamespace(
        publication_store=store,
        scan=SimpleNamespace(data_file="scan.nxs", gi=False),
    )
    h._rehydrate_publications_1d = MethodType(
        DisplayDataMixin._rehydrate_publications_1d, h)
    store.set_1d_hydrator(h._rehydrate_publications_1d)

    out = store.get_1d_many_or_hydrate((3, 4))

    assert calls == [("scan.nxs", (3, 4))]
    assert set(out) == {3, 4}
    assert out[3].view.has_1d
    assert out[3].view.intensity_2d is None
    assert out[3].view.raw is None
    assert out[3].raw_status == "1d-only"


def test_batch_1d_rehydrator_serializes_with_writer_file_lock(monkeypatch):
    import xrd_tools.io as xrd_io

    file_lock = threading.Lock()
    entered = threading.Event()

    def fake_get_1d(path, frame):
        entered.set()
        return _fake_1d_result(frame)

    monkeypatch.setattr(xrd_io, "get_1d", fake_get_1d)
    store = PublicationStore(max_heavy_items=0, max_thumbnail_items=0)
    h = SimpleNamespace(
        publication_store=store,
        scan=SimpleNamespace(data_file="scan.nxs", gi=False, file_lock=file_lock),
    )
    h._rehydrate_publications_1d = MethodType(
        DisplayDataMixin._rehydrate_publications_1d, h)

    file_lock.acquire()
    result = []
    thread = threading.Thread(
        target=lambda: result.extend(h._rehydrate_publications_1d((3, 4))))
    thread.start()
    try:
        assert not entered.wait(0.1), "bulk 1D read overlapped writer lock"
    finally:
        file_lock.release()
    thread.join(2.0)

    assert not thread.is_alive()
    assert entered.is_set()
    assert {publication.label for publication in result} == {3, 4}


def test_batch_1d_rehydrator_reads_default_chunks_under_file_lock(monkeypatch):
    import xrd_tools.io as xrd_io

    calls = []

    class _TrackingLock:
        held = False

        def __enter__(self):
            self.held = True
            return self

        def __exit__(self, *_exc):
            self.held = False
            return False

    file_lock = _TrackingLock()

    def fake_get_1d(path, frame):
        assert file_lock.held is True
        calls.append(tuple(frame))
        return _fake_1d_result(frame)

    monkeypatch.setattr(xrd_io, "get_1d", fake_get_1d)
    store = PublicationStore(max_heavy_items=0, max_thumbnail_items=0)
    h = SimpleNamespace(
        publication_store=store,
        scan=SimpleNamespace(data_file="scan.nxs", gi=False, file_lock=file_lock),
    )
    h._rehydrate_publications_1d = MethodType(
        DisplayDataMixin._rehydrate_publications_1d, h)

    out = h._rehydrate_publications_1d(range(600))

    assert [len(call) for call in calls] == [256, 256, 88]
    assert len(out) == 600


def test_writer_can_interleave_between_bulk_1d_hydration_chunks(monkeypatch):
    import xrd_tools.io as xrd_io

    class _FairLock:
        def __init__(self):
            self._cond = threading.Condition()
            self._held = False
            self._writer_pending = False
            self.writer_pending_seen = threading.Event()
            self.events = []

        def __enter__(self):
            with self._cond:
                while self._held or self._writer_pending:
                    self._cond.wait()
                self._held = True
                self.events.append("read-enter")
                return self

        def __exit__(self, *_exc):
            with self._cond:
                self.events.append("read-exit")
                self._held = False
                self._cond.notify_all()
                return False

        def writer_flush(self):
            with self._cond:
                self._writer_pending = True
                self.writer_pending_seen.set()
                self._cond.notify_all()
                while self._held:
                    self._cond.wait()
                self.events.append("writer-enter")
                self._held = True
                self._writer_pending = False
                self._cond.notify_all()
                self.events.append("writer-exit")
                self._held = False
                self._cond.notify_all()

    file_lock = _FairLock()
    writer_thread = []
    calls = []

    def fake_get_1d(path, frame):
        calls.append(tuple(frame))
        if len(calls) == 1:
            thread = threading.Thread(target=file_lock.writer_flush)
            writer_thread.append(thread)
            thread.start()
            assert file_lock.writer_pending_seen.wait(1.0)
        return _fake_1d_result(frame)

    monkeypatch.setattr(xrd_io, "get_1d", fake_get_1d)
    store = PublicationStore(max_heavy_items=0, max_thumbnail_items=0)
    h = SimpleNamespace(
        publication_store=store,
        _bulk_1d_read_chunk=2,
        scan=SimpleNamespace(data_file="scan.nxs", gi=False, file_lock=file_lock),
    )
    h._rehydrate_publications_1d = MethodType(
        DisplayDataMixin._rehydrate_publications_1d, h)

    out = h._rehydrate_publications_1d((1, 2, 3, 4, 5))
    writer_thread[0].join(2.0)

    assert not writer_thread[0].is_alive()
    assert [len(call) for call in calls] == [2, 2, 1]
    assert {publication.label for publication in out} == {1, 2, 3, 4, 5}
    assert file_lock.events[:5] == [
        "read-enter",
        "read-exit",
        "writer-enter",
        "writer-exit",
        "read-enter",
    ]


def test_request_frame_hydration_respects_enabled_flag():
    calls = []
    fake_worker = SimpleNamespace(
        request=lambda label, gen, *, purpose="full": calls.append(
            (label, gen, purpose)))
    h = SimpleNamespace(
        _async_hydration_enabled=False, display_generation=5,
        _hydration_pending_labels=set(),
    )
    h._ensure_hydration_worker = lambda: fake_worker
    h._request_frame_hydration = MethodType(
        displayFrameWidget._request_frame_hydration, h)

    h._request_frame_hydration(3)
    assert calls == []                        # disabled -> no background work

    h._async_hydration_enabled = True
    h._request_frame_hydration(3)
    assert calls == [(3, 5, "full")]          # enabled -> queued with generation
    h._request_frame_hydration(3)
    assert calls == [(3, 5, "full")]          # duplicate label/gen coalesced
    h._request_frame_hydration(3, purpose="1d")
    assert calls == [
        (3, 5, "full"),
        (3, 5, "1d"),
    ]                                        # same label, different purpose


def test_repeated_failed_hydration_suppresses_until_generation_bump(caplog):
    calls = []
    rendered = []
    fake_worker = SimpleNamespace(
        request=lambda label, gen, *, purpose="full": calls.append(
            (label, gen, purpose)))
    h = SimpleNamespace(
        _async_hydration_enabled=True,
        display_generation=5,
        _hydration_pending_labels=set(),
        _hydration_failure_counts={},
        _hydration_failure_logged=set(),
        _pending_hydration_render=False,
        _pending_hydration_generation=None,
    )
    h.update = lambda: rendered.append(True)
    h._ensure_hydration_worker = lambda: fake_worker
    h._request_frame_hydration = MethodType(
        displayFrameWidget._request_frame_hydration, h)
    h._flush_hydration_render = MethodType(
        displayFrameWidget._flush_hydration_render, h)
    h._on_frame_hydrated = MethodType(
        displayFrameWidget._on_frame_hydrated, h)

    with caplog.at_level(logging.DEBUG):
        for _ in range(5):
            before = len(calls)
            h._request_frame_hydration(8, purpose="1d")
            if len(calls) > before:
                h._on_frame_hydrated(8, 5)

    assert calls == [(8, 5, "1d")] * 3
    assert caplog.text.count("suppressing repeated hydration requests") == 1
    assert not [record for record in caplog.records if record.levelno >= logging.INFO]

    h.display_generation = 6
    h._request_frame_hydration(8, purpose="1d")
    assert calls[-1] == (8, 6, "1d")


def test_overlay_1d_pending_marker_does_not_block_retry():
    calls = []
    fake_worker = SimpleNamespace(
        request=lambda label, gen, *, purpose="full", **kwargs: calls.append(
            (label, gen, purpose, kwargs.get("consumer"))))
    h = SimpleNamespace(
        _async_hydration_enabled=True,
        display_generation=5,
        _hydration_pending_labels={(8, "1d", ConsumerKind.OVERLAY_1D.value)},
        _hydration_failure_counts={},
        ui=SimpleNamespace(
            plotMethod=SimpleNamespace(currentText=lambda: "Overlay")
        ),
    )
    h._ensure_hydration_worker = lambda: fake_worker
    h._request_frame_hydration = MethodType(
        displayFrameWidget._request_frame_hydration, h)

    h._request_frame_hydration(8, purpose="1d")

    assert calls == [(8, 5, "1d", ConsumerKind.OVERLAY_1D)]


def test_legacy_1d_catchup_requests_1d_hydration_purpose():
    calls = []
    def request(idx, *, purpose="full"):
        calls.append((idx, purpose))

    h = SimpleNamespace(
        idxs_1d=[],
        _snapshot_data=lambda idxs, allow_blocking_read=None: {7: (None, None)},
        _display_hydration_should_block=lambda allow_blocking_read=None: False,
        _hydrate_frame_from_disk=lambda idx, *, allow_blocking_read=True: None,
        _request_missing_publication=request,
    )

    ydata, xdata = DisplayDataMixin.get_frames_int_1d(h, [7])

    assert (ydata, xdata) == (None, None)
    assert calls == [(7, "1d")]


def test_legacy_2d_catchup_keeps_full_hydration_purpose():
    calls = []
    def request(idx, *, purpose="full"):
        calls.append((idx, purpose))

    h = SimpleNamespace(
        idxs_2d=[],
        _snapshot_data=lambda idxs, allow_blocking_read=None: {7: (None, None)},
        _display_hydration_should_block=lambda allow_blocking_read=None: False,
        _hydrate_frame_from_disk=lambda idx, *, allow_blocking_read=True: None,
        _request_missing_publication=request,
    )

    intensity, xdata, ydata = DisplayDataMixin.get_frames_int_2d(h, [7])

    assert (intensity, xdata, ydata) == (None, None, None)
    assert calls == [(7, "full")]


def test_on_frame_hydrated_discards_all_pending_purposes_for_label():
    rendered = []
    h = SimpleNamespace(
        display_generation=7,
        _hydration_pending_labels={(4, "full"), (4, "1d"), (5, "full")},
        _pending_hydration_render=False,
        _pending_hydration_generation=None,
    )
    h.update = lambda: rendered.append(True)
    h._flush_hydration_render = MethodType(
        displayFrameWidget._flush_hydration_render, h)
    h._on_frame_hydrated = MethodType(
        displayFrameWidget._on_frame_hydrated, h)

    h._on_frame_hydrated(4, 7)

    assert h._hydration_pending_labels == {(5, "full")}
    assert rendered == [True]


def test_on_frame_hydrated_drops_stale_generation():
    rendered = []
    h = SimpleNamespace(display_generation=7)
    h.update = lambda: rendered.append(True)
    h._flush_hydration_render = MethodType(
        displayFrameWidget._flush_hydration_render, h)
    h._on_frame_hydrated = MethodType(
        displayFrameWidget._on_frame_hydrated, h)

    h._on_frame_hydrated(4, 6)                 # stale (gen 6 != 7) -> dropped
    assert rendered == []
    h._on_frame_hydrated(4, 7)                 # current -> re-render
    assert rendered == [True]


def test_on_frame_hydrated_stale_overlay_1d_queues_append_repaint():
    requests = []
    h = SimpleNamespace(
        display_generation=8,
        _hydration_pending_labels={(4, "1d", ConsumerKind.OVERLAY_1D.value)},
        _overlay_hydrated_pending_append_labels=deque(),
        _pending_hydration_render=False,
        _pending_hydration_generation=None,
    )
    h.request_current_selection_repaint = (
        lambda *, generation=None, reason=None:
            requests.append((generation, reason)) or True
    )
    h._sync_selection_generation = lambda: 8
    h._hydration_purpose_resident = lambda label, purpose: True
    h._record_hydration_completion = (
        lambda label, purpose, *, success, generation: None
    )
    h._flush_hydration_render = MethodType(
        displayFrameWidget._flush_hydration_render, h)
    h._on_frame_hydrated = MethodType(
        displayFrameWidget._on_frame_hydrated, h)

    h._on_frame_hydrated(4, 7)

    assert list(h._overlay_hydrated_pending_append_labels) == [4]
    assert h._hydration_pending_labels == set()
    assert requests == [(8, "hydration")]


def test_on_frame_hydrated_stale_overlay_after_reset_is_cancelled():
    requests = []
    h = SimpleNamespace(
        display_generation=8,
        _hydration_pending_labels=set(),   # reset/scan-switch cleared it
        _overlay_hydrated_pending_append_labels=deque(),
        _pending_hydration_render=False,
        _pending_hydration_generation=None,
    )
    h.request_current_selection_repaint = (
        lambda *, generation=None, reason=None:
            requests.append((generation, reason)) or True
    )
    h._sync_selection_generation = lambda: 8
    h._flush_hydration_render = MethodType(
        displayFrameWidget._flush_hydration_render, h)
    h._on_frame_hydrated = MethodType(
        displayFrameWidget._on_frame_hydrated, h)

    h._on_frame_hydrated(4, 7)

    assert list(h._overlay_hydrated_pending_append_labels) == []
    assert requests == []


def test_on_frame_hydrated_accepts_batched_1d_labels():
    rendered = []
    h = SimpleNamespace(
        display_generation=7,
        _hydration_pending_labels={(1, "1d"), (2, "1d"), (3, "1d")},
        _pending_hydration_render=False,
    )
    h.update = lambda: rendered.append(True)
    h._flush_hydration_render = MethodType(
        displayFrameWidget._flush_hydration_render, h)
    h._on_frame_hydrated = MethodType(
        displayFrameWidget._on_frame_hydrated, h)

    h._on_frame_hydrated((1, 2), 7)
    assert h._hydration_pending_labels == {(3, "1d")}
    assert rendered == [True]


def test_hydration_completion_requests_current_selection_repaint():
    requests = []
    rendered = []
    h = SimpleNamespace(
        display_generation=7,
        _hydration_pending_labels={(4, "full")},
        _pending_hydration_render=False,
        _pending_hydration_generation=None,
    )
    h.update = lambda: rendered.append(True)
    h.request_current_selection_repaint = (
        lambda *, generation=None, reason=None:
            requests.append((generation, reason)) or True
    )
    h._flush_hydration_render = MethodType(
        displayFrameWidget._flush_hydration_render, h)
    h._on_frame_hydrated = MethodType(
        displayFrameWidget._on_frame_hydrated, h)

    h._on_frame_hydrated(4, 7)

    assert h._hydration_pending_labels == set()
    assert requests == [(7, "hydration")]
    assert rendered == []                    # completion never plots its label


def test_hydration_completion_drops_when_selection_changed_before_update():
    requests = []
    h = SimpleNamespace(
        display_generation=7,
        _last_selection_sig=((1,), False),
        frame_ids=["2"],
        overall=False,
        _hydration_pending_labels={(1, "full")},
        _pending_hydration_render=False,
        _pending_hydration_generation=None,
    )
    h.request_current_selection_repaint = (
        lambda *, generation=None, reason=None:
            requests.append((generation, reason)) or True
    )
    h._bump_display_generation = MethodType(
        displayFrameWidget._bump_display_generation, h)
    h._selection_generation_signature = MethodType(
        displayFrameWidget._selection_generation_signature, h)
    h._sync_selection_generation = MethodType(
        displayFrameWidget._sync_selection_generation, h)
    h._flush_hydration_render = MethodType(
        displayFrameWidget._flush_hydration_render, h)
    h._on_frame_hydrated = MethodType(
        displayFrameWidget._on_frame_hydrated, h)

    h._on_frame_hydrated(1, 7)

    assert h.display_generation == 8
    assert requests == []
    assert h._pending_hydration_render is False


def test_hydration_completion_stream_coalesces_rerenders():
    rendered = []

    class FakeTimer:
        def __init__(self):
            self.starts = 0

        def start(self):
            self.starts += 1

    quiet = FakeTimer()
    progress = FakeTimer()
    h = SimpleNamespace(
        display_generation=7,
        _hydration_pending_labels={(label, "full") for label in range(100)},
        _pending_hydration_render=False,
        _last_hydration_render=0.0,
        _hydration_quiet_timer=quiet,
        _hydration_progress_timer=progress,
    )
    h.update = lambda: rendered.append(True)
    h._flush_hydration_render = MethodType(
        displayFrameWidget._flush_hydration_render, h)
    h._flush_hydration_progress_render = MethodType(
        displayFrameWidget._flush_hydration_progress_render, h)
    h._on_frame_hydrated = MethodType(
        displayFrameWidget._on_frame_hydrated, h)

    for label in range(50):
        h._on_frame_hydrated(label, 7)

    assert rendered == []
    assert quiet.starts == 50
    assert progress.starts == 50
    h._flush_hydration_progress_render()
    assert rendered == [True]

    for label in range(50, 100):
        h._on_frame_hydrated(label, 7)

    assert len(rendered) == 1
    h._flush_hydration_render()
    assert len(rendered) == 2


def test_full_raw_thumbnail_only_hydration_self_suppresses_rl1():
    """RL-1 run-end treadmill: the last frame's full raw was evicted during live;
    only a THUMBNAIL persisted.  The render re-requests purpose="full"; the
    hydration "completes" with only the thumbnail resident.  The OLD
    _view_has_hydration_payload counted a thumbnail as satisfying "full" ->
    success=True -> the failure count never reached the limit ->
    _hydration_request_suppressed never tripped -> the run-end display re-requested
    "full" forever.  After the tier-accurate fix a thumbnail does NOT satisfy
    "full" -> success=False -> suppression trips after <=3 completions -> bounded.

    Spins-before / terminates-after: on pre-fix code len(calls) reaches the loop
    bound (8, unbounded); after the fix it stops at _HYDRATION_FAILURE_LIMIT (3).
    """
    from xdart.gui.tabs.static_scan.display_frame_widget import (
        _HYDRATION_FAILURE_LIMIT)
    calls = []
    fake_worker = SimpleNamespace(
        request=lambda label, gen, *, purpose="full", **kw: calls.append(
            (label, gen, purpose)))
    # a store item whose view has a THUMBNAIL but NO full raw and NO 2d payload
    view = SimpleNamespace(
        raw=None, thumbnail=np.zeros((2, 2), dtype=float),
        has_2d=False, intensity_2d=None, has_1d=False, intensity_1d=None)
    item = SimpleNamespace(view=view)
    store = SimpleNamespace(get=lambda label: item)
    h = SimpleNamespace(
        _async_hydration_enabled=True,
        display_generation=5,
        _hydration_pending_labels=set(),
        _hydration_failure_counts={},
        _hydration_failure_logged=set(),
        _pending_hydration_render=False,
        _pending_hydration_generation=None,
        _hydration_stores=lambda: (store,),
    )
    h.update = lambda: None
    h._ensure_hydration_worker = lambda: fake_worker
    for name in ("_request_frame_hydration", "_flush_hydration_render",
                 "_on_frame_hydrated", "_hydration_request_suppressed",
                 "_hydration_purpose_resident", "_view_has_hydration_payload",
                 "_hydration_item_views", "_record_hydration_completion"):
        setattr(h, name, MethodType(getattr(displayFrameWidget, name), h))

    LABEL = 3621
    # Simulate the render->hydrate->complete->repaint->render cycle: request the
    # full raw, and whenever a worker request is actually issued, drive its
    # completion (as _on_frame_hydrated would on sigHydrated).
    for _ in range(8):
        before = len(calls)
        h._request_frame_hydration(LABEL, purpose="full")
        if len(calls) > before:
            h._on_frame_hydrated(LABEL, 5)

    assert len(calls) <= _HYDRATION_FAILURE_LIMIT, (
        f"treadmill: full-raw hydration re-requested {len(calls)}x unbounded "
        "(thumbnail-only mis-scored as a successful full hydration)")
    assert h._hydration_request_suppressed(LABEL, "full") is True
    # a further render-driven re-request issues NO new worker request
    before = len(calls)
    h._request_frame_hydration(LABEL, purpose="full")
    assert len(calls) == before, "suppressed request must not reach the worker"


def test_resident_tier_hydration_request_is_skipped_rl1_overlay():
    """RL-1 OVERLAY driver (distinct from the Single thumbnail-mis-scoring bug):
    the render re-issues _request_frame_hydration for the current frame on every
    repaint even when its tier is ALREADY resident (the last frame at run-end
    whose full raw IS present).  The worker "completes" it success=True ->
    schedules a hydration repaint -> re-request -> a treadmill the failure-count
    backoff CANNOT stop (each completion is a SUCCESS).  A resident tier must be
    skipped so NO worker request is issued.
    """
    calls = []
    fake_worker = SimpleNamespace(
        request=lambda label, gen, *, purpose="full", **kw: calls.append(
            (label, gen, purpose)))
    view = SimpleNamespace(                       # full raw RESIDENT
        raw=np.zeros((2, 2)), thumbnail=np.zeros((2, 2)),
        has_2d=True, intensity_2d=np.zeros((2, 2)),
        has_1d=True, intensity_1d=np.zeros(2))
    item = SimpleNamespace(view=view)
    store = SimpleNamespace(get=lambda label: item)
    h = SimpleNamespace(
        _async_hydration_enabled=True, display_generation=7,
        _hydration_pending_labels=set(), _hydration_failure_counts={},
        _hydration_stores=lambda: (store,))
    h._ensure_hydration_worker = lambda: fake_worker
    for name in ("_request_frame_hydration", "_hydration_request_suppressed",
                 "_hydration_purpose_resident", "_view_has_hydration_payload",
                 "_hydration_item_views"):
        setattr(h, name, MethodType(getattr(displayFrameWidget, name), h))

    # Drive the render loop: request "full" each cycle, clearing pending between
    # (as _on_frame_hydrated's completion does).  Pre-fix this issues one worker
    # request per cycle (unbounded); after the resident-guard it issues NONE.
    for _ in range(5):
        h._request_frame_hydration(651, purpose="full")
        h._hydration_pending_labels.clear()
    assert calls == [], \
        "a resident tier must not be re-requested (RL-1 overlay treadmill)"


def test_succeeded_frame_not_re_requested_after_eviction_rl1_show_all():
    """RL-1 Show All: a frame that hydrated SUCCESSFULLY must not be re-requested
    after the store EVICTS it (Show All selects more frames than the store cap, so
    a hydrated frame soon reads non-resident again).  Without the success-set guard
    this is a success-treadmill neither the failure backoff (it's a success) nor
    the resident-guard (it's non-resident by then) can stop; the plot renders such
    frames from the bulk path anyway, so ONE successful store hydration per
    generation suffices.
    """
    calls = []
    fake_worker = SimpleNamespace(
        request=lambda label, gen, *, purpose="full", **kw: calls.append(
            (label, gen, purpose)))
    view = SimpleNamespace(
        raw=None, thumbnail=None, has_2d=False, intensity_2d=None,
        has_1d=False, intensity_1d=None)
    item = SimpleNamespace(view=view)
    store = SimpleNamespace(get=lambda label: item)
    h = SimpleNamespace(
        _async_hydration_enabled=True, display_generation=3,
        _hydration_pending_labels=set(), _hydration_failure_counts={},
        _hydration_failure_logged=set(),
        _pending_hydration_render=False, _pending_hydration_generation=None,
        _hydration_stores=lambda: (store,))
    h.update = lambda: None
    h._ensure_hydration_worker = lambda: fake_worker
    for name in ("_request_frame_hydration", "_flush_hydration_render",
                 "_on_frame_hydrated", "_hydration_request_suppressed",
                 "_hydration_purpose_resident", "_view_has_hydration_payload",
                 "_hydration_item_views", "_record_hydration_completion"):
        setattr(h, name, MethodType(getattr(displayFrameWidget, name), h))

    LABEL = 3600
    # cycle 1: not resident -> request issued
    h._request_frame_hydration(LABEL, purpose="1d")
    assert calls == [(LABEL, 3, "1d")]
    view.has_1d = True                              # hydration made it resident
    h._on_frame_hydrated(LABEL, 3)                  # SUCCESS -> success-set
    view.has_1d = False                             # store evicts it (cap pressure)
    for _ in range(5):                              # render loop re-requests it
        h._hydration_pending_labels.clear()
        h._request_frame_hydration(LABEL, purpose="1d")
    assert calls == [(LABEL, 3, "1d")], \
        "an already-succeeded frame must not be re-requested after eviction"
    # a selection change (new generation) allows a fresh request
    h.display_generation = 4
    h._request_frame_hydration(LABEL, purpose="1d")
    assert calls[-1] == (LABEL, 4, "1d")
