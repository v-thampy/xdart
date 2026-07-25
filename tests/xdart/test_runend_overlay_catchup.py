"""Run-end overlay catch-up (PERF-3 Option A) — outcome + guard tests.

Two layers:

* **Production-wired outcome tests** — a REAL ``staticWidget`` (offscreen), a
  REAL small ``.nxs`` written through the production writer, the publication
  store resident for only the head of the scan, and the REAL
  ``wrangler_finished`` LIVE saw-frames branch driven end-to-end.  The positive
  test asserts the OUTCOME the failed Item-2 test never did: the waterfall
  accumulator reaches ``set(scan.frames.index)``, with the tail servable ONLY
  via the disk/_LoadFramesWorker path (non-resident-tail trap).  ``show_all``
  is NOT faked in the positive test; negative tests may spy on it to detect a
  call that must not happen.
* **Guard/decode unit tests** — the catch-up decision logic and the
  length-tolerant row-id decode (slice-mode rows are 3-tuples; a literal
  ``(skey, fidx)`` unpack ValueErrors) on a controlled host, the direct
  class-method-call pattern this repo uses for run-end delegates.
"""
from __future__ import annotations

import gc
import logging
import threading
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
from xrd_tools.session.display_logic import (
    frame_index_from_qualified_id,
    scan_key_from_qualified_id,
)

N_FRAMES = 40
N_TAIL = 8          # non-resident tail: on disk, never published to the store
N_Q = 64
N_CHI = 16
RADIAL = np.linspace(0.5, 5.0, N_Q, dtype=np.float32)
AZIM = np.linspace(-180.0, 180.0, N_CHI, endpoint=False, dtype=np.float32)
SCAN_NAME = "catchup_scan"


@pytest.fixture(scope="module")
def qapp():
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    try:
        yield app
    finally:
        for top in list(app.topLevelWidgets()):
            try:
                top.close()
                top.deleteLater()
            except Exception:
                pass
        for _ in range(5):
            app.processEvents()


# ── real .nxs fixture (production writer) ────────────────────────────────────
#
# Minimal duck-typed LiveFrame/LiveScan for save_scan_to_nexus — the writer
# only reads attribute trees off frames (same pattern as
# test_nexus_writer_roundtrip.py, duplicated here so this module does not
# inherit that module's nexusformat importorskip).

class _DuckPONI:
    def __init__(self):
        self.dist = 0.1
        self.poni1 = 0.05
        self.poni2 = 0.05
        self.rot1 = 0.0
        self.rot2 = 0.0
        self.rot3 = 0.0


class _DuckFrame:
    def __init__(self, idx):
        rng = np.random.default_rng(idx)
        self.idx = int(idx)
        self.poni = _DuckPONI()
        self.source_file = f"frame_{idx:04d}.tif"
        self.source_frame_idx = 0
        self.skip_map_raw = True
        self.map_raw = None
        self.bg_raw = 0
        self.mask = None
        # One shared radial grid for the WHOLE test module: the store-resident
        # head and the disk-loaded tail must land on one compatible overlay
        # grid key, or the accumulator would reset mid-catch-up.
        self.int_1d = SimpleNamespace(
            radial=RADIAL,
            intensity=np.full(N_Q, float(idx), dtype=np.float32),
            sigma=rng.random(N_Q, dtype=np.float32) * 0.1,
            unit="q_A^-1",
        )
        self.int_2d = SimpleNamespace(
            radial=RADIAL,
            azimuthal=AZIM,
            intensity=rng.random((N_Q, N_CHI), dtype=np.float32),
            unit="q_A^-1",
            azimuthal_unit="deg",
            sigma=None,
        )
        self.gi_1d = {}
        self.gi_2d = {}
        self.thumbnail = rng.random((16, 16), dtype=np.float32)


class _DuckFrames:
    def __init__(self, frames):
        self._by_idx = {f.idx: f for f in frames}
        self.index = [f.idx for f in frames]
        self._in_memory = self._by_idx

    def __iter__(self):
        return (self._by_idx[i] for i in self.index)

    def __getitem__(self, idx):
        return self._by_idx[idx]

    def __len__(self):
        return len(self.index)


class _DuckScan:
    def __init__(self, frames):
        self.frames = _DuckFrames(frames)
        self.scan_data = pd.DataFrame(
            {"tth": np.linspace(10.0, 14.0, len(frames), dtype=np.float32)})
        self.bai_1d_args = {"numpoints": N_Q}
        self.bai_2d_args = {"npt_rad": N_Q, "npt_azim": N_CHI}
        self.mg_args = {"wavelength": 1.0e-10}
        self.geometry = None
        self.global_mask = None
        self.detector_shape = None
        self.gi = False
        self.skip_2d = False
        self.stitched_1d = None
        self.stitched_2d = None


@pytest.fixture(scope="module")
def catchup_nxs(tmp_path_factory):
    """A real 40-frame .nxs (idx 1..40) written through the production writer."""
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus
    scan = _DuckScan([_DuckFrame(i) for i in range(1, N_FRAMES + 1)])
    path = tmp_path_factory.mktemp("catchup") / f"{SCAN_NAME}.nxs"
    save_scan_to_nexus(scan, path, mode="w", finalize=False)
    return path


@pytest.fixture
def widget(qapp):
    """A real staticWidget, torn down after each test."""
    for _ in range(3):
        qapp.processEvents()
    w = staticWidget()
    try:
        yield w
    finally:
        try:
            w.close()
        except Exception:
            pass
        try:
            w.deleteLater()
        except Exception:
            pass
        for _ in range(3):
            qapp.processEvents()
        gc.collect()
        for _ in range(2):
            qapp.processEvents()


def _store_frame(i):
    """A light 1D-only frame on the SAME grid as the .nxs, for store residency."""
    return SimpleNamespace(
        idx=i,
        int_1d=SimpleNamespace(
            radial=RADIAL,
            intensity=np.full(N_Q, float(i), dtype=np.float32),
            sigma=np.ones(N_Q, dtype=np.float32),
            unit="q_A^-1",
        ),
        int_2d=None, map_raw=None, mask=None, gi=False, gi_2d={},
        thumbnail=None, bg_raw=0, scan_info={},
        source_file=f"frame_{i:04d}.tif", source_frame_idx=0)


def _publish_head(w, upto):
    """Make frames 1..upto store-resident through the production publish seam."""
    from xdart.modules.frame_publication import publication_from_live_frame
    store = w.publication_store
    for i in range(1, upto + 1):
        store.upsert(publication_from_live_frame(
            _store_frame(i), generation=store.generation,
            # X1 3c: mirror the production publish stamp (explicit scan owner).
            scan_key=SCAN_NAME))
    return store


def _wire_live_run_end(w, monkeypatch, nxs):
    """Point the real widget at the written scan and stub ONLY the run
    externals that don't exist offscreen (same set as the production run-end
    wiring tests in test_gui_modes_end_to_end.py).  wrangler_finished,
    integrator_thread_finished, clear_overlay, the reconcile, show_all and the
    load/render machinery all stay REAL."""
    w.scan.name = SCAN_NAME
    w.displayframe.ui.plotMethod.setCurrentText("Overlay")
    w.h5viewer.auto_last = True
    monkeypatch.setattr(w.wrangler.thread, "batch_mode", False, raising=False)
    monkeypatch.setattr(w.wrangler.thread, "xye_only", False, raising=False)
    monkeypatch.setattr(w.wrangler.thread, "fname", str(nxs), raising=False)
    monkeypatch.setattr(w.wrangler, "scan_name", SCAN_NAME, raising=False)
    monkeypatch.setattr(w.integratorTree.integrator_thread, "isRunning",
                        lambda: False)
    monkeypatch.setattr(w.wrangler, "stop", lambda: None)
    w.h5viewer.dirname = str(nxs.parent)
    w._enter_run_state()
    w._run_saw_frame = True                      # the LIVE saw-frames branch


def _pump(qapp, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.02)


def _history_ids(w):
    hist = getattr(w.displayframe, "_waterfall_history", None)
    return tuple(getattr(hist, "ids", ()) or ())


def _decoded_indices(w):
    return {frame_index_from_qualified_id(r) for r in _history_ids(w)
            if scan_key_from_qualified_id(r) == w.scan.name}


def _quiet(w):
    v = w.h5viewer
    return (getattr(v, "_load_worker", None) is None
            and not v._selection_coalesce_timer.is_pending()
            and not v._load_coalesce_timer.is_pending()
            and not v._update_coalesce_timer.is_pending())


# ── production-wired outcome tests ───────────────────────────────────────────

def test_disk_loaded_single_to_overlay_retains_six_rapid_visits(
        qapp, widget, catchup_nxs):
    """The Single seed and five disk-backed arrow taps all accumulate."""
    from PySide6 import QtCore, QtGui

    w = widget
    w.scan.set_datafile(str(catchup_nxs), name=SCAN_NAME)
    v = w.h5viewer
    v.auto_last = False
    v.new_scan_loaded = False
    v.update_data(force_rebuild=True)
    lw = v.ui.listData
    lw.setCurrentRow(0, QtCore.QItemSelectionModel.ClearAndSelect)

    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.02)
        if _quiet(w) and w.displayframe.plot_data[1].shape[0] == 1:
            break

    w.displayframe.ui.plotMethod.setCurrentText("Overlay")
    qapp.processEvents()
    assert _history_ids(w) == ((SCAN_NAME, 1),)

    press = QtGui.QKeyEvent(
        QtCore.QEvent.Type.KeyPress,
        QtCore.Qt.Key.Key_Down,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    release = QtGui.QKeyEvent(
        QtCore.QEvent.Type.KeyRelease,
        QtCore.Qt.Key.Key_Down,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    for row in range(1, 6):
        assert v.eventFilter(lw, press) is False
        lw.setCurrentRow(row, QtCore.QItemSelectionModel.ClearAndSelect)
        qapp.processEvents()
        assert v.eventFilter(lw, release) is False
        time.sleep(0.19)
        qapp.processEvents()

    deadline = time.monotonic() + 8.0
    expected = tuple((SCAN_NAME, label) for label in range(1, 7))
    while time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.02)
        if _quiet(w) and _history_ids(w) == expected:
            break

    assert _history_ids(w) == expected

    w.displayframe.clear_1D()
    assert w.displayframe._waterfall_history is None
    assert list(w.displayframe._overlay_hydrated_pending_append_labels) == []

    v.show_all()
    all_rows = tuple((SCAN_NAME, label) for label in range(1, N_FRAMES + 1))
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.02)
        if _quiet(w) and _history_ids(w) == all_rows:
            break

    assert _history_ids(w) == all_rows


def test_runend_catchup_recovers_non_resident_tail_from_disk(
        qapp, widget, monkeypatch, catchup_nxs, caplog):
    """THE outcome test Item-2 lacked: after a live Overlay run whose tail
    display payloads were never published (store-resident head only, tail on
    disk), the run-end catch-up must drive the accumulator to the FULL frame
    index — through the real one-shot -> quiescence -> show_all -> disk-load ->
    completion-render spine, with no fake on any seam."""
    w = widget
    store = _publish_head(w, N_FRAMES - N_TAIL)
    # The trap, pinned: the tail is NOT resident anywhere in memory — only the
    # .nxs can serve it.
    resident = {int(i) for i in store.labels() if isinstance(i, int)}
    assert resident == set(range(1, N_FRAMES - N_TAIL + 1))
    assert not w.viewer_rows_1d and not w.viewer_rows_2d

    _wire_live_run_end(w, monkeypatch, catchup_nxs)
    caplog.set_level(logging.INFO)

    w.wrangler_finished()

    # Armed by the LIVE branch (token = scan name), frame index reconciled
    # from the written file.
    assert w._runend_catchup_token == SCAN_NAME
    assert {int(i) for i in w.scan.frames.index} == set(range(1, N_FRAMES + 1))

    expected = {int(i) for i in w.scan.frames.index}
    counts = []
    deadline = time.monotonic() + 12.0
    while time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.02)
        counts.append(len(_history_ids(w)))
        if (w._runend_catchup_token is None and _quiet(w)
                and _decoded_indices(w) == expected):
            break
    # Let any trailing coalesced render land, still tracking monotonicity.
    settle = time.monotonic() + 0.5
    while time.monotonic() < settle:
        qapp.processEvents()
        time.sleep(0.02)
        counts.append(len(_history_ids(w)))

    assert w._runend_catchup_token is None, "catch-up never fired/consumed"
    assert _decoded_indices(w) == expected, (
        f"accumulator short: {sorted(_decoded_indices(w))} != all "
        f"{N_FRAMES} frames")
    # The accumulator must never shrink once the catch-up is armed (a reset
    # mid-catch-up would repeat the Item-2 failure shape).
    assert all(b >= a for a, b in zip(counts, counts[1:])), (
        f"accumulator row count decreased after arming: {counts}")
    # One-shot: at most one fire.  On a fast machine the browser's own
    # post-live disk load may complete the tail before this callback runs; in
    # that valid race the catch-up must take its idempotent already-complete
    # path instead of issuing a redundant Show All.
    fires = [r for r in caplog.records if "-> show_all()" in r.getMessage()]
    assert len(fires) <= 1
    if fires:
        assert fires[0].levelno == logging.INFO
    else:
        assert any("already complete, no-op" in r.getMessage()
                   for r in caplog.records)
    # The tail became resident via the disk load (not some in-memory source).
    assert set(range(N_FRAMES - N_TAIL + 1, N_FRAMES + 1)) <= {
        int(i) for i in store.labels() if isinstance(i, int)}


def test_user_frame_click_before_callback_cancels(
        qapp, widget, monkeypatch, catchup_nxs):
    """A user gesture between arming and the callback must cancel the catch-up.
    The production gesture signal is the frame click (listData.itemClicked ->
    disable_auto_last); the committed fix deliberately does NOT gate on a bare
    display_generation compare — the run-end selection-collapse echoes bump it
    programmatically, which is exactly what killed Item-2's deferred render."""
    w = widget
    _publish_head(w, N_FRAMES - N_TAIL)
    _wire_live_run_end(w, monkeypatch, catchup_nxs)

    w.wrangler_finished()
    assert w._runend_catchup_token == SCAN_NAME

    calls = []
    monkeypatch.setattr(w.h5viewer, "show_all", lambda: calls.append(1))
    # The real click wire: reconcile rebuilt listData, click its first frame.
    item = w.h5viewer.ui.listData.item(0)
    assert item is not None
    w.h5viewer.ui.listData.itemClicked.emit(item)
    assert w.h5viewer.auto_last is False

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and w._runend_catchup_token is not None:
        qapp.processEvents()
        time.sleep(0.02)
    _pump(qapp, 0.5)
    assert w._runend_catchup_token is None
    assert calls == [], "show_all fired after a user gesture"


def test_single_mode_run_end_is_a_noop(qapp, widget, monkeypatch, catchup_nxs):
    w = widget
    _publish_head(w, N_FRAMES - N_TAIL)
    _wire_live_run_end(w, monkeypatch, catchup_nxs)
    w.displayframe.ui.plotMethod.setCurrentText("Single")

    calls = []
    monkeypatch.setattr(w.h5viewer, "show_all", lambda: calls.append(1))
    w.wrangler_finished()

    assert w._runend_catchup_token is None       # never armed
    _pump(qapp, 1.0)
    assert calls == []


def test_new_run_clears_pending_token(qapp, widget, monkeypatch, catchup_nxs):
    """_enter_run_state supersedes a pending catch-up from the previous run."""
    w = widget
    _publish_head(w, N_FRAMES - N_TAIL)
    _wire_live_run_end(w, monkeypatch, catchup_nxs)

    w.wrangler_finished()
    assert w._runend_catchup_token == SCAN_NAME

    calls = []
    monkeypatch.setattr(w.h5viewer, "show_all", lambda: calls.append(1))
    w._enter_run_state()                         # next run starts immediately
    assert w._runend_catchup_token is None
    _pump(qapp, 1.5)
    assert calls == []
    w._exit_run_state(w._new_projection_receipt())


def test_new_file_clears_pending_token(qapp, widget, monkeypatch, catchup_nxs):
    """The set_file/data_reset cascade (sigNewFile ->
    _on_new_file_display_reset) supersedes a pending catch-up — a post-run file
    open must never be followed by a surprise show_all."""
    w = widget
    _publish_head(w, N_FRAMES - N_TAIL)
    _wire_live_run_end(w, monkeypatch, catchup_nxs)

    w.wrangler_finished()
    assert w._runend_catchup_token == SCAN_NAME

    calls = []
    monkeypatch.setattr(w.h5viewer, "show_all", lambda: calls.append(1))
    w.h5viewer.sigNewFile.emit(str(catchup_nxs))  # the set_datafile signal
    assert w._runend_catchup_token is None
    _pump(qapp, 1.5)
    assert calls == []


def test_already_complete_accumulator_is_idempotent(
        qapp, widget, monkeypatch, catchup_nxs):
    """Guard 3 on a REAL, fully-populated accumulator: nothing missing -> the
    callback consumes the token without calling show_all.  The accumulator is
    built by the real show_all/render spine first (everything store-resident),
    then the callback is invoked directly — deterministic, no race against the
    run-end echo cascade."""
    w = widget
    _publish_head(w, N_FRAMES)                   # fully resident
    w.scan.name = SCAN_NAME
    w.displayframe.ui.plotMethod.setCurrentText("Overlay")
    w.h5viewer.auto_last = True
    assert w.scan.load_frame_index_only(str(catchup_nxs)) == N_FRAMES

    w.h5viewer.show_all()                        # real machinery fills history
    deadline = time.monotonic() + 10.0
    expected = set(range(1, N_FRAMES + 1))
    while time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.02)
        if _quiet(w) and _decoded_indices(w) == expected:
            break
    assert _decoded_indices(w) == expected

    calls = []
    monkeypatch.setattr(w.h5viewer, "show_all", lambda: calls.append(1))
    w._runend_catchup_token = SCAN_NAME          # as the arm would set it
    w._runend_catchup_tries = 0
    staticWidget._runend_overlay_catchup(w)
    assert calls == []
    assert w._runend_catchup_token is None       # consumed as a no-op
    assert _decoded_indices(w) == expected


def test_never_quiescent_gives_up_quietly_after_bounded_tries(
        qapp, widget, monkeypatch, catchup_nxs, caplog):
    """Quiescence never reached: the deadline bounds polling and reports that
    the displayed waterfall may be incomplete."""
    w = widget
    _publish_head(w, N_FRAMES - N_TAIL)
    _wire_live_run_end(w, monkeypatch, catchup_nxs)
    caplog.set_level(logging.DEBUG)
    w._runend_catchup_timeout_s = 1.0

    w.wrangler_finished()
    assert w._runend_catchup_token == SCAN_NAME
    n_records_at_arm = len(caplog.records)

    calls = []
    monkeypatch.setattr(w.h5viewer, "show_all", lambda: calls.append(1))
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline and w._runend_catchup_token is not None:
        # Debounce restart faster than its 100 ms interval -> never quiet.
        w.h5viewer._load_coalesce_timer.start()
        qapp.processEvents()
        time.sleep(0.02)

    assert w._runend_catchup_token is None, "never gave up"
    assert w._runend_catchup_tries >= 2
    assert calls == []
    loud = [r for r in caplog.records[n_records_at_arm:]
            if "catch-up timed out" in r.getMessage()]
    assert len(loud) == 1
    _pump(qapp, 1.0)                             # let the held-back load settle


def test_batch_run_end_does_not_arm(qapp, widget, monkeypatch, catchup_nxs):
    """The hook is the LIVE saw-frames branch only: a batch run ends through
    its own reload + select-last recovery and must not arm the catch-up."""
    w = widget
    _wire_live_run_end(w, monkeypatch, catchup_nxs)
    monkeypatch.setattr(w.wrangler.thread, "batch_mode", True, raising=False)

    calls = []
    monkeypatch.setattr(w.h5viewer, "show_all", lambda: calls.append(1))
    w.wrangler_finished()
    assert w._runend_catchup_token is None
    _pump(qapp, 1.0)
    assert calls == []


# ── guard / decode unit tests (controlled host) ──────────────────────────────

def _q(scan_key, idx, projection=None):
    """A scan-qualified row id: 2-tuple normally, 3-tuple in slice mode."""
    return (scan_key, idx) if projection is None else (scan_key, idx, projection)


def _host(*, index, have_ids, method="Overlay", auto_last=True,
          scan_name="test_scan", token="test_scan", busy=False):
    show_all_calls = []
    host = SimpleNamespace(
        scan=SimpleNamespace(
            name=scan_name,
            scan_lock=threading.Lock(),
            frames=SimpleNamespace(index=list(index)),
        ),
        displayframe=SimpleNamespace(
            # current_scan_key(displayframe) reads displayframe.scan.name
            scan=SimpleNamespace(name=scan_name),
            _waterfall_history=SimpleNamespace(ids=list(have_ids)),
            display_generation=0,
            ui=SimpleNamespace(
                plotMethod=SimpleNamespace(currentText=lambda: method)),
        ),
        h5viewer=SimpleNamespace(
            auto_last=auto_last,
            show_all=lambda: show_all_calls.append(1),
            _load_worker=(object() if busy else None),
            _browse_one_shot_pending_render=False,
        ),
        _runend_catchup_token=token,
        _runend_catchup_tries=0,
        _runend_catchup_deadline=time.monotonic() + 60.0,
    )
    return host, show_all_calls


def test_missing_ids_basic():
    host, _ = _host(index=[1, 2, 3, 4, 5],
                    have_ids=[_q("test_scan", i) for i in (1, 2, 3)])
    assert staticWidget._runend_overlay_missing_ids(host) == {4, 5}


def test_missing_ids_handles_slice_mode_3tuples():
    # BLOCKING bug the verification caught: slice-mode rows are 3-tuples
    # (scan_key, frame_idx, projection_id); a literal 2-tuple unpack ValueErrors.
    host, _ = _host(
        index=[1, 2, 3, 4],
        have_ids=[_q("test_scan", 1, "qz"), _q("test_scan", 2, "qz")])
    assert staticWidget._runend_overlay_missing_ids(host) == {3, 4}  # no ValueError


def test_missing_ids_ignores_other_scans():
    host, _ = _host(index=[1, 2, 3],
                    have_ids=[_q("OTHER", 1), _q("OTHER", 2), _q("test_scan", 1)])
    assert staticWidget._runend_overlay_missing_ids(host) == {2, 3}


def test_missing_ids_cleared_accumulator_is_all_missing():
    host, _ = _host(index=[1, 2, 3], have_ids=[])
    host.displayframe._waterfall_history = None   # clear_overlay wiped it
    assert staticWidget._runend_overlay_missing_ids(host) == {1, 2, 3}


def test_catchup_fires_show_all_when_missing_and_quiescent():
    host, calls = _host(index=[1, 2, 3, 4, 5],
                        have_ids=[_q("test_scan", i) for i in (1, 2, 3)])
    staticWidget._runend_overlay_catchup(host)
    assert calls == [1]                          # fired exactly once
    assert host._runend_catchup_token is None    # token consumed


def test_catchup_skips_when_complete():
    host, calls = _host(index=[1, 2, 3],
                        have_ids=[_q("test_scan", i) for i in (1, 2, 3)])
    staticWidget._runend_overlay_catchup(host)
    assert calls == []
    assert host._runend_catchup_token is None


@pytest.mark.parametrize("kwargs", [
    dict(auto_last=False),          # user clicked a frame -> disable_auto_last
    dict(method="Single"),          # not an overlay mode
    dict(token="OLD_scan"),         # scan.name changed since arming
    dict(token=None),               # cleared by a new run (_enter_run_state)
])
def test_catchup_skips_on_cancellation(kwargs):
    host, calls = _host(index=[1, 2, 3, 4],
                        have_ids=[_q("test_scan", 1)], **kwargs)
    staticWidget._runend_overlay_catchup(host)
    assert calls == []


def test_catchup_rearms_while_busy_then_gives_up(qapp):
    host, calls = _host(index=[1, 2, 3, 4],
                        have_ids=[_q("test_scan", 1)], busy=True)
    # Busy callbacks retain ownership until the absolute deadline.
    for _ in range(12):
        staticWidget._runend_overlay_catchup(host)
        assert calls == []
        assert host._runend_catchup_token == "test_scan"
    host._runend_catchup_deadline = time.monotonic() - 1.0
    staticWidget._runend_overlay_catchup(host)
    assert calls == []
    assert host._runend_catchup_token is None


def test_numeric_container_suffixes_remain_distinct_run_owners():
    assert staticWidget._canonical_run_token("sample_00005.nxs") == \
        "sample_00005"
    assert staticWidget._canonical_run_token("sample_00006.nxs") == \
        "sample_00006"
    assert staticWidget._canonical_run_token("sample_00005_master.h5") == \
        "sample_00005"


def test_stale_autofit_generation_cannot_touch_a_new_run(monkeypatch):
    calls = []
    host = SimpleNamespace(
        _runend_generation=3,
        _runend_autofit_generation=2,
        _run_active=True,
    )
    monkeypatch.setattr(
        staticWidget,
        "_runend_waterfall_autofit",
        lambda self: calls.append("fit"),
    )

    staticWidget._runend_autofit_when_quiet(
        host, generation=2, deadline=time.monotonic() + 10.0)

    assert calls == []


ALIAS_N = 651
ALIAS_TAIL = 19          # non-resident cadence-shaped tail (on disk only)


@pytest.fixture(scope="module")
def alias_651_nxs(tmp_path_factory):
    """A real 651-frame .nxs with the CANONICAL output name (production
    writer), for the Eiger ``_master`` alias run-end case (H18-R8)."""
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus
    scan = _DuckScan([_DuckFrame(i) for i in range(1, ALIAS_N + 1)])
    path = tmp_path_factory.mktemp("alias651") / f"{SCAN_NAME}.nxs"
    save_scan_to_nexus(scan, path, mode="w", finalize=False)
    return path


def _canonical_key(name):
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        _scan_key_from_source,
    )
    return _scan_key_from_source(str(name)) or str(name)


def _decoded_canonical(w):
    target = _canonical_key(SCAN_NAME)
    return {frame_index_from_qualified_id(r) for r in _history_ids(w)
            if _canonical_key(scan_key_from_qualified_id(r)) == target}


def test_eiger_master_alias_arms_and_completes_651_frames(
        qapp, widget, monkeypatch, alias_651_nxs, caplog):
    """H18-R8: the widget owner retains the pre-canonical ``*_master`` scan
    spelling while the worker/output canonicalized it — run ownership must
    resolve through the canonical run identity, shared finalization and the
    one-shot catch-up must still run exactly once, and catch-up completion is
    an IDENTITY outcome: the accumulator's scan-qualified set equals the
    authoritative finished frame index (651/651, no reset), through the
    normal coalescing path."""
    w = widget
    _publish_head(w, ALIAS_N - ALIAS_TAIL)
    _wire_live_run_end(w, monkeypatch, alias_651_nxs)
    # THE ALIAS: widget owner keeps the pre-canonical Eiger spelling.
    w.scan.name = f"{SCAN_NAME}_master"
    caplog.set_level(logging.INFO)

    # H18-R8(b) preface: a mid-run PROGRAMMATIC crop (mode projection /
    # cadence render) — NOT a genuine user zoom — must not survive run end.
    # A finished 651-frame Overlay renders as a Waterfall, so the plot the
    # run-end auto-fit acts on is ``wf_widget.image_plot`` (that IS this
    # displayframe's ``_active_bottom_plot()`` once >15 traces are shown).
    # Crop THAT plot's x-axis: capturing ``_active_bottom_plot()`` before the
    # run returns the LINE plot (the Waterfall is not active yet), which the
    # run-end auto-fit never touches — so a crop on it could never be observed
    # as re-armed and the assertion below would be vacuous (H18-E1).
    wf_plot = getattr(getattr(w.displayframe, "wf_widget", None),
                      "image_plot", None)
    assert wf_plot is not None, \
        "the Waterfall image plot must exist for the 651-frame Overlay run"
    wf_plot.setRange(xRange=(0.0, 3.0), padding=0)     # disables x auto-range
    assert wf_plot.getViewBox().autoRangeEnabled()[0] is False, \
        "the programmatic crop must disable Waterfall x auto-range pre-run-end"

    w.wrangler_finished()

    assert w._runend_catchup_token is not None, \
        "the aliased owner must still arm the run-end catch-up (H18-R8)"
    arms = [r for r in caplog.records
            if "run-end overlay catch-up: armed" in r.getMessage()]
    assert len(arms) == 1

    expected = {int(i) for i in w.scan.frames.index}
    assert expected == set(range(1, ALIAS_N + 1))
    counts = []
    deadline = time.monotonic() + 25.0
    while time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.02)
        counts.append(len(_history_ids(w)))
        if (w._runend_catchup_token is None and _quiet(w)
                and _decoded_canonical(w) == expected):
            break
    settle = time.monotonic() + 0.6
    while time.monotonic() < settle:
        qapp.processEvents()
        time.sleep(0.02)
        counts.append(len(_history_ids(w)))

    assert w._runend_catchup_token is None, "catch-up never fired/consumed"
    missing = expected - _decoded_canonical(w)
    assert not missing, (
        f"accumulator identity short of the finished index: "
        f"{len(missing)} missing (e.g. {sorted(missing)[:8]})")
    assert all(b >= a for a, b in zip(counts, counts[1:])), \
        f"accumulator shrank after arming (reset): {counts}"
    fires = [r for r in caplog.records if "-> show_all()" in r.getMessage()]
    assert len(fires) <= 1, "catch-up must fire at most once"

    # H18-R8(b): full-history auto-fit after the automatic run-end reseed.
    # No genuine user zoom was recorded, so the mid-run x crop must NOT
    # survive: the NORMAL run-end path (show_all -> _runend_autofit_when_quiet
    # -> _runend_waterfall_autofit, never called directly here) re-arms x
    # auto-range on the finished Waterfall plot.  Assert the CROPPED axis
    # specifically — the untouched y stays auto-ranged regardless, so the old
    # ``auto[0] or auto[1]`` could not tell a working auto-fit from a no-op
    # (the H18-E1 vacuity).  pyqtgraph reports an enabled axis as a float
    # (1.0) and a disabled one as ``False``; the auto-fit is async
    # (QTimer.singleShot ~250 ms after show_all), so poll to an explicit
    # outcome rather than sleeping a fixed interval.
    wf_vb = wf_plot.getViewBox()
    autofit_deadline = time.monotonic() + 8.0
    while (time.monotonic() < autofit_deadline
           and wf_vb.autoRangeEnabled()[0] is False):
        qapp.processEvents()
        time.sleep(0.02)
    assert wf_vb.autoRangeEnabled()[0] is not False, \
        "run-end reseed must re-arm the cropped Waterfall x auto-range when " \
        "no genuine user zoom is recorded (H18-R8(b))"


def test_runend_autofit_respects_genuine_user_zoom(qapp, widget):
    """H18-R8(b) negative: a recorded genuine user zoom is preserved."""
    w = widget
    w.displayframe._wf_user_zoomed = True
    plot = w.displayframe._active_bottom_plot()
    if plot is None:
        pytest.skip("no active bottom plot in this configuration")
    plot.setRange(xRange=(1.0, 2.0), padding=0)
    staticWidget._runend_waterfall_autofit(w)
    auto = plot.getViewBox().autoRangeEnabled()
    assert auto[0] is False, \
        "a genuine user zoom must survive the run-end auto-fit (x stays pinned)"


def test_genuine_zoom_during_run_is_recorded_via_real_signal(qapp, widget):
    """H18-R12: the manual-range listener must be connected from RUN START
    (not first at run end), on BOTH switchable bottom plots, so a zoom made
    during processing survives the run-end auto-fit."""
    w = widget
    w._enter_run_state()
    df = w.displayframe
    assert getattr(df, "_wf_user_zoomed", None) is False

    plots = {df.plot}
    active = df._active_bottom_plot()
    if active is not None:
        plots.add(active)
    assert plots, "no bottom plots to test"
    for plot in plots:
        df._wf_user_zoomed = False
        vb = plot.getViewBox()
        vb.sigRangeChangedManually.emit(vb.autoRangeEnabled())
        assert df._wf_user_zoomed is True, \
            f"manual range signal on {plot!r} was not recorded (H18-R12)"

    # ...and the recorded zoom blocks the run-end auto-fit
    target = active if active is not None else df.plot
    target.setRange(xRange=(1.0, 2.0), padding=0)
    staticWidget._runend_waterfall_autofit(w)
    assert target.getViewBox().autoRangeEnabled()[0] is False
    w._exit_run_state(w._new_projection_receipt())


def test_waterfall_viewbox_hooked_at_run_entry_below_threshold(qapp, widget):
    """H18-R14: an Overlay run that starts BELOW the Waterfall threshold must
    still have the concrete Waterfall view box connected at run entry, so a
    genuine manual-range gesture after acquisition crosses the threshold is
    recorded."""
    w = widget
    w.displayframe.ui.plotMethod.setCurrentText("Overlay")
    df = w.displayframe
    assert not df._waterfall_active()          # below the 15-trace threshold

    w._enter_run_state()
    assert df._wf_user_zoomed is False

    wf_plot = df.wf_widget.image_plot          # the concrete Waterfall plot
    vb = wf_plot.getViewBox()
    vb.sigRangeChangedManually.emit(vb.autoRangeEnabled())
    assert df._wf_user_zoomed is True, \
        "the Waterfall view box must be hooked at run entry even while the " \
        "line plot is the active bottom plot (H18-R14)"
    w._exit_run_state(w._new_projection_receipt())
