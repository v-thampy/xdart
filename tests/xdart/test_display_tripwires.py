# -*- coding: utf-8 -*-
"""QW-4 tripwires.

1. accumulate_waterfall skip-on-mix: a cross-unit batch that cannot
   canonicalize (no carried per-row wavelength, D1) is skipped (never
   np.interp'd across disjoint domains — the constant-clamp/blank-band
   failure), with an ERROR naming the units.  PRODUCTION behavior since V1
   Stage 3 (previously XDART_DEBUG_DISPLAY-gated); since Stage 4 the skip
   applies regardless of grid size — the legacy same-size relabel branch is
   DELETED (a display unit flip converts at draw and never reaches the
   accumulator).  Flag-on and flag-off paths are pinned below.
2. GUI-thread file_lock assert (XDART_DEBUG_DISPLAY=1 only): the GUI thread
   blocking-acquiring the writer-coordinating lock while a run is active is
   the BB-1 beachball class; the tripwire logs ERROR with the acquiring
   stack.
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


def test_cross_unit_lambda_less_is_skipped_under_debug(monkeypatch, caplog):
    monkeypatch.setenv("XDART_DEBUG_DISPLAY", "1")
    hist = _hist(unit="q_A^-1", n=128)
    # Incoming λ-less batch in a DIFFERENT unit on a DIFFERENT-size grid:
    # without the tripwire these rows would interp across disjoint domains
    # and append clamped bands.
    x2 = np.linspace(10.0, 55.0, 200)
    with caplog.at_level(logging.ERROR):
        out = accumulate_waterfall(
            hist, reset_key="grid", unit="2th_deg", x=x2,
            rows=[np.linspace(5.0, 6.0, 200)], ids=[("A", 1)], names=["A/1"])
    assert list(out.ids) == [("A", 0)]          # batch skipped, nothing lost
    assert np.asarray(out.rows).shape[0] == 1
    assert any("cross-unit" in r.message for r in caplog.records)


def test_cross_unit_lambda_less_skips_in_production_flag_off(
        monkeypatch, caplog):
    # V1 Stage 3: the skip IS production behavior (no env gate).  A λ-less
    # cross-unit batch is dropped for this render with an ERROR instead of
    # being appended unconverted — the conscious flip of the pre-V1
    # "appended (known hazard)" default, per the canonical-grid plan's D1
    # policy.
    monkeypatch.delenv("XDART_DEBUG_DISPLAY", raising=False)
    hist = _hist(unit="q_A^-1", n=128)
    x2 = np.linspace(10.0, 55.0, 200)
    with caplog.at_level(logging.ERROR):
        out = accumulate_waterfall(
            hist, reset_key="grid", unit="2th_deg", x=x2,
            rows=[np.linspace(5.0, 6.0, 200)], ids=[("A", 1)], names=["A/1"])
    assert list(out.ids) == [("A", 0)]          # batch skipped, nothing lost
    assert np.asarray(out.rows).shape[0] == 1
    # The emitted history keeps its OWN unit — the axis never lies about
    # unconverted values.
    assert out.unit == "q_A^-1"
    assert any("cross-unit" in r.message for r in caplog.records)


def test_cross_unit_same_size_lambda_less_also_skips(monkeypatch, caplog):
    # V1 Stage 4: the legacy same-size relabel branch is DELETED (pre-Stage-4
    # this exact call relabelled the grid in place and appended).  A λ-less
    # cross-unit batch now skips regardless of grid size — with or without
    # the debug flag; the carried-λ path canonicalizes instead (D1, pinned in
    # test_display_logic).
    monkeypatch.setenv("XDART_DEBUG_DISPLAY", "1")
    n = 128
    hist = _hist(unit="q_A^-1", n=n)
    x2 = np.linspace(10.0, 55.0, n)             # same size: used to relabel
    with caplog.at_level(logging.ERROR):
        out = accumulate_waterfall(
            hist, reset_key="grid", unit="2th_deg", x=x2,
            rows=[np.linspace(5.0, 6.0, n)], ids=[("A", 1)], names=["A/1"])
    assert list(out.ids) == [("A", 0)]          # batch skipped, nothing lost
    assert out.unit == "q_A^-1"                 # the axis never lies
    assert np.allclose(out.x, np.linspace(1.0, 5.0, n))   # grid untouched
    assert any("cross-unit" in r.message for r in caplog.records)


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
