# -*- coding: utf-8 -*-
"""MEM-1[14] follow-up: the per-label publication memo must be LIVE on the
REAL widget.

The original memo (9bcbc9c5) was test-green but prod-dead: its identity
lookup queried ``_active_frame_record_store`` on the displayframe — a
staticWidget attribute that does not exist there — so the identity was
always ``(None, None)`` and the memo never engaged (memo_hits=+0
memo_misses=+0 on every real tick).  These tests drive the REAL
staticWidget -> displayframe -> record-store wiring end to end and assert
the memo's observable outcomes: a repeat resolve is a HIT, and republishing
the record (new identity) invalidates.  If the wiring ever regresses to a
name the displayframe doesn't have, the hit assertion fails immediately.
"""
import gc

import pytest
from pyqtgraph.Qt import QtWidgets

from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
from xdart.modules.frame_publication import publication_from_live_frame
from xrd_tools.session.frame_record_store import FrameRecordStore

from .test_frame_publication import DuckFrame


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture()
def widget(qapp):
    w = staticWidget()
    try:
        yield w
    finally:
        w.close()
        w.deleteLater()
        for _ in range(3):
            qapp.processEvents()
        gc.collect()


def _install_record(widget, idx):
    """Put a REAL FrameRecord for ``idx`` into the widget's active record
    store (creating the store on first use), the same store
    ``store_first_frame_view`` resolves."""
    records = widget._active_frame_record_store()
    if records is None:
        records = FrameRecordStore(max_items=None, max_heavy_items=None)
        widget._frame_record_store = records
    pub = publication_from_live_frame(DuckFrame(idx=idx))
    assert pub.record is not None
    records.upsert(pub.record)
    return records


def test_memo_engages_on_the_real_widget(widget):
    _install_record(widget, 0)
    df = widget.displayframe

    hits0 = getattr(df, "_pub_memo_hits", 0)
    misses0 = getattr(df, "_pub_memo_misses", 0)

    first = df._store_first_publication_for_display(0)
    assert first is not None, "store-first path did not resolve the record"
    assert getattr(df, "_pub_memo_misses", 0) == misses0 + 1
    assert getattr(df, "_pub_memo_hits", 0) == hits0

    second = df._store_first_publication_for_display(0)
    assert second is not None
    assert getattr(df, "_pub_memo_hits", 0) == hits0 + 1, (
        "memo never engaged on the real widget — the identity lookup is "
        "prod-dead again (MEM-1[14] follow-up regression)")
    # Cheap re-stamp on a hit: same view contents, current generation.
    assert second.view is first.view


def test_memo_invalidates_when_the_record_is_republished(widget):
    records = _install_record(widget, 0)
    df = widget.displayframe

    assert df._store_first_publication_for_display(0) is not None
    hits = getattr(df, "_pub_memo_hits", 0)
    misses = getattr(df, "_pub_memo_misses", 0)

    # Re-publish label 0: a NEW record object (new identity) must MISS —
    # a hit here would serve a stale frame after re-publication/hydration.
    pub = publication_from_live_frame(DuckFrame(idx=0))
    records.upsert(pub.record)
    rebuilt = df._store_first_publication_for_display(0)
    assert rebuilt is not None
    assert getattr(df, "_pub_memo_misses", 0) == misses + 1
    assert getattr(df, "_pub_memo_hits", 0) == hits


def test_blocking_reads_bypass_the_memo(widget):
    _install_record(widget, 0)
    df = widget.displayframe

    assert df._store_first_publication_for_display(0) is not None
    hits = getattr(df, "_pub_memo_hits", 0)
    misses = getattr(df, "_pub_memo_misses", 0)

    # User gestures resolve fresh: neither counter moves.
    got = df._store_first_publication_for_display(0, allow_blocking_read=True)
    assert got is not None
    assert getattr(df, "_pub_memo_hits", 0) == hits
    assert getattr(df, "_pub_memo_misses", 0) == misses
