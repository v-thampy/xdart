# -*- coding: utf-8 -*-
"""QW-4 debug tripwires (XDART_DEBUG_DISPLAY=1 only; zero production cost).

1. accumulate_waterfall assert-on-mix: a cross-unit batch with the relabel
   NOT engaged must be skipped (never np.interp'd across disjoint domains —
   the constant-clamp/blank-band failure), with an ERROR naming the units.
2. GUI-thread file_lock assert: the GUI thread blocking-acquiring the
   writer-coordinating lock while a run is active is the BB-1 beachball
   class; the tripwire logs ERROR with the acquiring stack.
"""
import logging
import threading

import numpy as np
import pytest

from xrd_tools.session.display_logic import accumulate_waterfall


def _hist(unit="q_A^-1", n=128):
    x = np.linspace(1.0, 5.0, n)
    return accumulate_waterfall(
        None, reset_key="grid", unit=unit, x=x,
        rows=[np.linspace(1.0, 2.0, n)], ids=[("A", 0)], names=["A/0"])


def test_cross_unit_no_relabel_is_skipped_under_debug(monkeypatch, caplog):
    monkeypatch.setenv("XDART_DEBUG_DISPLAY", "1")
    hist = _hist(unit="q_A^-1", n=128)
    # Incoming batch in a DIFFERENT unit on a DIFFERENT-size grid: the unit
    # relabel cannot engage (sizes differ), so without the tripwire these
    # rows would interp across disjoint domains and append clamped bands.
    x2 = np.linspace(10.0, 55.0, 200)
    with caplog.at_level(logging.ERROR):
        out = accumulate_waterfall(
            hist, reset_key="grid", unit="2th_deg", x=x2,
            rows=[np.linspace(5.0, 6.0, 200)], ids=[("A", 1)], names=["A/1"])
    assert list(out.ids) == [("A", 0)]          # batch skipped, nothing lost
    assert np.asarray(out.rows).shape[0] == 1
    assert any("cross-unit" in r.message for r in caplog.records)


def test_cross_unit_no_relabel_legacy_when_flag_off(monkeypatch):
    monkeypatch.delenv("XDART_DEBUG_DISPLAY", raising=False)
    hist = _hist(unit="q_A^-1", n=128)
    x2 = np.linspace(10.0, 55.0, 200)
    out = accumulate_waterfall(
        hist, reset_key="grid", unit="2th_deg", x=x2,
        rows=[np.linspace(5.0, 6.0, 200)], ids=[("A", 1)], names=["A/1"])
    # Production behavior unchanged: the row is appended (the known hazard
    # the tripwire exists to expose; pinned here so flipping the default
    # someday is a conscious decision, not an accident).
    assert list(out.ids) == [("A", 0), ("A", 1)]


def test_unit_relabel_still_engages_under_debug(monkeypatch):
    # A clean Q<->2theta flip on the SAME grid size must still relabel and
    # keep appending — the tripwire must not break the OV contract.
    monkeypatch.setenv("XDART_DEBUG_DISPLAY", "1")
    n = 128
    hist = _hist(unit="q_A^-1", n=n)
    x2 = np.linspace(10.0, 55.0, n)             # same size -> relabel engages
    out = accumulate_waterfall(
        hist, reset_key="grid", unit="2th_deg", x=x2,
        rows=[np.linspace(5.0, 6.0, n)], ids=[("A", 1)], names=["A/1"])
    assert list(out.ids) == [("A", 0), ("A", 1)]
    assert np.allclose(out.x, x2)               # grid relabeled in place


def test_gui_thread_file_lock_tripwire_fires(monkeypatch, caplog, qapp):
    """Drive the REAL _locked_scan_read on the GUI thread of a real widget
    with a run flagged active: the tripwire must log the BB-1-class ERROR
    (and must stay silent from a worker thread / when idle)."""
    from xdart.gui.tabs.static_scan.display_data import DisplayDataMixin

    monkeypatch.setenv("XDART_DEBUG_DISPLAY", "1")

    class _Owner(DisplayDataMixin):
        pass

    owner = _Owner()
    owner.file_lock = threading.Condition()
    owner._processing_active = True

    with caplog.at_level(logging.ERROR):
        with owner._locked_scan_read():
            pass
    assert any("tripwire" in r.message for r in caplog.records), \
        "GUI-thread blocking-acquire under _processing_active did not trip"

    # Idle widget: silent.
    caplog.clear()
    owner._processing_active = False
    with caplog.at_level(logging.ERROR):
        with owner._locked_scan_read():
            pass
    assert not any("tripwire" in r.message for r in caplog.records)

    # Same acquire from a NON-GUI thread: silent (workers legitimately wait).
    caplog.clear()
    owner._processing_active = True
    err = []

    def _worker():
        try:
            with owner._locked_scan_read():
                pass
        except Exception as exc:  # pragma: no cover
            err.append(exc)

    t = threading.Thread(target=_worker)
    t.start()
    t.join(5)
    assert not err
    assert not any("tripwire" in r.message for r in caplog.records)


@pytest.fixture(scope="module")
def qapp():
    from pyqtgraph.Qt import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
