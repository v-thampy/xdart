"""Run-end overlay catch-up (PERF-3 Option A) — guard logic + missing-set decode.

Drives the REAL ``staticWidget`` catch-up methods against a controlled host (the
direct class-method-call pattern this repo uses for run-end delegates).  The
end-to-end OUTCOME (the accumulator reaching N via ``show_all``'s async disk
load) is gated on the live verification slot per the spec; here we lock down the
DECISION logic and the length-tolerant id decode — specifically the slice-mode
3-tuple ``ValueError`` the verification pass caught in the original spec formula.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget


@pytest.fixture(scope="module")
def qapp():
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


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
    )
    return host, show_all_calls


# ── missing-set decode ────────────────────────────────────────────────────────

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


# ── fire / no-fire decision ───────────────────────────────────────────────────

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
    # Each call while busy re-arms (bounded to 8), never fires.
    for _ in range(8):
        staticWidget._runend_overlay_catchup(host)
        assert calls == []
        assert host._runend_catchup_token == "test_scan"
    # 9th call: tries now exceeds 8 -> give up, clear the token, still no fire.
    staticWidget._runend_overlay_catchup(host)
    assert calls == []
    assert host._runend_catchup_token is None
