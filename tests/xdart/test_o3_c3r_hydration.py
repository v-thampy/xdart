# -*- coding: utf-8 -*-
"""X1 O-3 c3R-b — hydration target and commit ownership.

FROZEN BEFORE the c3R-b production edits (§9.4), red at the c3R-a tip.

The worker used to resolve its stores from a live provider at EXECUTION time,
so a request built under a browse could land in the acquisition's store; the
context check ran only when the completion returned to the GUI, after the
payload had already been inserted; and the dedupe key omitted the context, so
the same label in two sub-scans collapsed into one request.

The correction is that a request carries its OWN target and its own commit
authority: the disk read stays off-GUI and unlocked, and a short context-owned
gate around the final insertion is what invalidation, rescope and release
linearize against.

RULE 10: every invocation of this module must set ``XDART_SESSION_FILE`` to a
unique path under ``/Users/vthampy/repos/tmp``.
"""

from __future__ import annotations

import logging
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("pyqtgraph")
from pyqtgraph import Qt
from pyqtgraph.Qt import QtWidgets

from xdart.gui.tabs.static_scan.display_frame_widget import displayFrameWidget
from xdart.gui.tabs.static_scan.frame_hydration_worker import (
    FrameHydrationWorker,
)
from xdart.gui.tabs.static_scan.run_config_debug import (
    DECISION_HYDRATION_CONTEXT_MISMATCH,
)
from xdart.gui.tabs.static_scan.static_scan_widget import (
    RUN_ORIGIN_REINTEGRATE,
    staticWidget,
)
from xdart.modules.display_context import (
    BrowseContext,
    ContextKind,
    new_context_token,
)

_DIRECT = Qt.QtCore.Qt.ConnectionType.DirectConnection


class _Store:
    def __init__(self, name="store"):
        self.name = name
        self.cleared = 0

    def clear(self):
        self.cleared += 1


def _browse(**overrides):
    token = new_context_token(ContextKind.BROWSE)
    fields = dict(
        context_token=token,
        load_generation=1,
        operation=None,
        requested_path="/data/browse_b.nxs",
        scan_key="browse-b",
        scan=object(),
        frame=None,
        frame_ids=["1"],
        frames={},
        viewer_rows_1d={},
        viewer_rows_2d={},
        publication_store=_Store("B"),
    )
    fields.update(overrides)
    return BrowseContext(**fields)


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def widget(qapp, monkeypatch, tmp_path):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    monkeypatch.setenv("XDART_SESSION_FRESH", "1")
    monkeypatch.setenv("XDART_SESSION_FILE", str(tmp_path / "session.json"))
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget as sw

    value = sw()
    try:
        yield value
    finally:
        for name in ("_acquisition_context", "_browse_context",
                     "_display_selection"):
            try:
                setattr(value, name, None)
            except Exception:
                pass
        try:
            value._run_active = False
            value._controls_v2_refresh_timer.cancel()
        except Exception:
            pass
        value.close()
        value.deleteLater()
        qapp.processEvents()


def _browsing_run(widget, monkeypatch, tmp_path):
    """A real paused run with a real admitted browse selected."""
    widget._enter_run_state(origin=RUN_ORIGIN_REINTEGRATE)
    widget.h5viewer.paused_browse_active = True
    queued = []
    monkeypatch.setattr(widget.h5viewer, "_ensure_file_thread_running",
                        lambda: None)
    monkeypatch.setattr(widget.h5viewer.file_thread.queue, "put",
                        queued.append)
    context = widget._begin_paused_browse(str(tmp_path / "browsed.nxs"))
    widget._on_browse_loaded(queued[0])
    assert widget._display_selection.names(context)
    return context


# --------------------------------------------------------------------------- #
# §9.2 — the commit gate
# --------------------------------------------------------------------------- #

def test_a_cancelled_commit_gate_admits_no_insertion():
    """The gate is the linearization point between read and insert.

    A read already in flight may FINISH READING — that work is off-GUI and
    unlocked — but once its context has been invalidated it may insert into
    nothing at all.
    """
    context = _browse()
    gate = context.commit_gate
    assert gate.epoch == context.commit_epoch

    assert gate.enter(gate.epoch) is True
    gate.leave()

    context.invalidate()
    assert gate.enter(gate.epoch) is False, (
        "an invalidated context still authorised a store insertion")
    # A stale epoch is refused even before any cancellation.
    fresh = _browse()
    assert fresh.commit_gate.enter(fresh.commit_gate.epoch - 1) is False


def test_the_commit_gate_never_holds_its_lock_across_a_read():
    """Cancellation must not be able to wait on disk I/O.

    ``cancel()`` runs on the GUI thread at Resume and at run end.  If the gate
    were held across the read, that call would block the GUI for the duration
    of an ``.nxs`` open — the multi-second freeze this whole machinery exists
    to remove.  The gate is therefore only ever held around the INSERT.
    """
    context = _browse()
    gate = context.commit_gate
    cancelled = threading.Event()

    def cancel_from_another_thread():
        context.invalidate()
        cancelled.set()

    # While NOT inside the gate (i.e. during the read), cancel proceeds at once.
    thread = threading.Thread(target=cancel_from_another_thread)
    thread.start()
    thread.join(2.0)
    assert cancelled.is_set(), "cancellation blocked outside the commit window"
    assert gate.enter(gate.epoch) is False


def test_same_context_token_does_not_authorize_a_foreign_scan_key():
    """Reviewer discriminator 2, promoted verbatim in intent."""
    host = SimpleNamespace(
        display_context_token="context-A",
        scan=SimpleNamespace(name="scan-new", data_file=""),
    )

    assert displayFrameWidget._admit_hydration_owner(
        host, ("context-A", "scan-old"), 7, 3) is False

def test_request_hydrates_the_request_time_store_not_the_later_selection():
    """Reviewer discriminator 3, promoted verbatim in intent."""
    entered = threading.Event()
    release = threading.Event()
    complete = threading.Event()
    calls = {"A": [], "B": []}

    class Store:
        def __init__(self, name):
            self.name = name

        def get_or_hydrate(self, label, **_kwargs):
            calls[self.name].append(label)
            if label == 1:
                entered.set()
                assert release.wait(5.0)
            return {"label": label}

    stores = {"current": (Store("B"),)}
    worker = FrameHydrationWorker(lambda: stores["current"])
    worker.sigHydrated.connect(
        lambda label, _generation, _owner: (
            complete.set() if label == 2 else None),
        _DIRECT)
    worker.start()
    try:
        worker.request(1, 3, context_token="context-B",
                       context_scan_key="scan-B")
        assert entered.wait(5.0)
        worker.request(2, 3, context_token="context-B",
                       context_scan_key="scan-B")
        stores["current"] = (Store("A"),)
        release.set()
        assert complete.wait(5.0)
    finally:
        release.set()
        worker.stop()

    assert calls == {"A": [], "B": [1, 2]}

def test_same_label_generation_in_two_scan_keys_is_not_deduped():
    """Reviewer discriminator 4, promoted verbatim in intent."""
    calls = []
    done = threading.Event()

    class Store:
        def get_or_hydrate(self, label, **_kwargs):
            calls.append(label)
            if len(calls) == 2:
                done.set()
            return {"label": label}

    worker = FrameHydrationWorker(Store())
    worker.request(7, 3, context_token="context-A",
                   context_scan_key="scan-one")
    worker.request(7, 3, context_token="context-A",
                   context_scan_key="scan-two")
    worker.start()
    try:
        assert done.wait(2.0)
    finally:
        worker.stop()

    assert calls == [7, 7]


# --------------------------------------------------------------------------- #
# §9.2.8 — the two frozen race positions, on the real widget
# --------------------------------------------------------------------------- #

def _store_state(widget, browse):
    return {
        "a_labels": tuple(sorted(widget.publication_store.snapshot())),
        "b_labels": tuple(sorted(browse.publication_store.snapshot())),
        "display_scan": widget.displayframe.scan,
        "selection": widget._display_selection,
    }


def test_a_request_made_under_b_never_lands_in_a_after_the_selection_moves(
        widget, monkeypatch, tmp_path):
    """§9.2.8 position 1 — queued under B, selection moves before resolution.

    The request must carry its target; resolving it from the display at
    execution time is how B's read landed in A's store.
    """
    browse = _browsing_run(widget, monkeypatch, tmp_path)
    display = widget.displayframe
    b_stores = display._hydration_stores()
    assert browse.publication_store in b_stores

    request = display._build_hydration_request(3, purpose="full")
    assert request is not None, "no owned hydration request was built"
    assert request.context_token == browse.context_token
    assert request.context_scan_key == browse.scan_key
    assert browse.publication_store in request.stores
    assert widget.publication_store not in request.stores

    before = _store_state(widget, browse)
    # The selection moves to A before the request is ever resolved.
    widget.h5viewer.paused_browse_active = False
    staticWidget._select_acquisition_context(widget, origin="resume")
    staticWidget._invalidate_browse_context(widget, reason="resume")

    # The request's target is unchanged — it is a value, not a lookup.
    assert browse.publication_store in request.stores
    assert widget.publication_store not in request.stores
    # And its commit authority is gone, so it may insert into nothing.
    assert request.commit_gate.enter(request.epoch) is False

    after = _store_state(widget, browse)
    assert after["a_labels"] == before["a_labels"]
    assert after["b_labels"] == before["b_labels"]


def test_a_read_that_returns_after_resume_inserts_into_neither_context(
        widget, monkeypatch, tmp_path, caplog):
    """§9.2.8 position 2 — the disk read returns AFTER Resume invalidated B.

    Section 4's consequence, restored: the invalidated request changes neither
    context, and produces exactly one correlated rejection.
    """
    import logging

    from xdart.gui.tabs.static_scan.run_config_debug import (
        DECISION_HYDRATION_CONTEXT_MISMATCH,
    )

    monkeypatch.setenv("XDART_RUN_CONFIG_DEBUG", "1")
    caplog.set_level(logging.INFO)
    browse = _browsing_run(widget, monkeypatch, tmp_path)
    display = widget.displayframe

    request = display._build_hydration_request(5, purpose="full")
    assert request is not None
    before = _store_state(widget, browse)

    # The read has already started; Resume lands while it is in flight.
    widget.h5viewer.paused_browse_active = False
    staticWidget._select_acquisition_context(widget, origin="resume")
    staticWidget._invalidate_browse_context(widget, reason="resume")

    # The read returns now.  It may not insert anywhere.
    assert request.commit_gate.enter(request.epoch) is False
    admitted = displayFrameWidget._admit_hydration_owner(
        display, (request.context_token, request.context_scan_key),
        5, request.generation)
    assert admitted is False

    after = _store_state(widget, browse)
    assert after["a_labels"] == before["a_labels"]
    assert after["b_labels"] == before["b_labels"]
    assert after["display_scan"] is widget.scan

    rejections = [
        record for record in caplog.records
        if DECISION_HYDRATION_CONTEXT_MISMATCH in record.getMessage()]
    assert len(rejections) == 1, (
        f"expected exactly one correlated rejection, got {len(rejections)}")


def test_an_active_context_never_produces_an_ownerless_request(
        widget, monkeypatch, tmp_path):
    """§9.2.1/9.2.5 — fail closed on missing, malformed and empty owners.

    A run installs its acquisition selection at admission, so a production
    request made while a context is active always carries an owner.
    """
    widget._enter_run_state(origin=RUN_ORIGIN_REINTEGRATE)
    display = widget.displayframe
    assert widget._display_selection is not None, (
        "Run admission installed no acquisition DisplaySelection")
    assert display.display_context_token == \
        widget._acquisition_context.context_token

    for owner in (None, (), ("",), ("", ""), "not-a-pair", ("tok",),
                  (None, None)):
        assert displayFrameWidget._admit_hydration_owner(
            display, owner, 1, 0) is False, owner

    good = (display.display_context_token,
            widget._acquisition_context.scan_key)
    assert displayFrameWidget._admit_hydration_owner(
        display, good, 1, 0) is True


def test_a_rescope_restamps_the_acquisition_selection(
        widget, monkeypatch, tmp_path):
    """§9.2.2 — a genuine sub-scan boundary re-stamps the request identity.

    Without this a Directory rescope moved ``current_scan_key`` while the
    selection kept naming the previous sub-scan, so every later request was
    qualified against a key the display had already left.
    """
    widget._enter_run_state(origin=RUN_ORIGIN_REINTEGRATE)
    acquisition = widget._acquisition_context
    before = widget._display_selection
    assert before.scan_key == acquisition.scan_key

    widget._rescope_frame_panel_to("run-a-sub2")
    assert acquisition.scan_key == "run-a-sub2"
    after = widget._display_selection
    assert after is not before, "the selection was not restamped"
    assert after.scan_key == "run-a-sub2"
    assert after.context_token == acquisition.context_token
    assert widget.displayframe.display_context_token == \
        acquisition.context_token


# --------------------------------------------------------------------------- #
# §10.1 — the REAL FrameRecordStore under the commit gate
# --------------------------------------------------------------------------- #

def _record_store_with_evicted_frame():
    import numpy as np
    from xrd_tools.core import FrameRecord, FrameView, IntegrationResult1D
    from xrd_tools.session import FrameRecordStore

    def view(label, scale):
        intensity = scale * np.array([3.0, 6.0, 12.0])
        return FrameView.from_results(
            label=label,
            result_1d=IntegrationResult1D(
                radial=np.array([0.1, 0.2, 0.3]),
                intensity=intensity, sigma=np.sqrt(intensity),
                unit="q_A^-1"))

    store = FrameRecordStore(max_heavy_items=1)
    original = view(1, 1.0)
    store.upsert(FrameRecord.from_view(original), persisted=True)
    store.upsert(FrameRecord.from_view(view(2, 2.0)), persisted=True)
    assert not store.has_heavy_payload(1)
    return store, original


def test_the_real_record_store_hydrates_under_a_live_gate():
    """§10.1 — the owned `purpose="2d"` consumer, on the real store.

    `_hydration_stores()` captures BOTH stores and c3R-b sends the gate to
    every one of them, but only `PublicationStore` accepted it; the worker
    swallowed the resulting `TypeError` as a generic hydration failure, so an
    evicted integrated frame reported completion and never came back.
    """
    from xrd_tools.core import FrameRecord
    from xdart.modules.display_context import CommitGate

    store, original = _record_store_with_evicted_frame()
    store.set_hydrator(lambda label: FrameRecord.from_view(original))
    gate = CommitGate()

    assert store.get_or_hydrate(
        1, commit_gate=gate, commit_epoch=gate.epoch) is not None
    assert store.has_heavy_payload(1)


def test_the_real_record_store_still_hydrates_without_a_gate():
    """§10.1.1 — the no-gate API is preserved for idle and legacy callers."""
    from xrd_tools.core import FrameRecord

    store, original = _record_store_with_evicted_frame()
    store.set_hydrator(lambda label: FrameRecord.from_view(original))
    assert store.get_or_hydrate(1) is not None
    assert store.has_heavy_payload(1)


def test_a_gate_cancelled_while_the_record_hydrator_is_latched_inserts_nothing():
    """§10.1.3 — a cancelled gate inserts NOTHING and moves no bookkeeping.

    It must not fall back to an ungated upsert, and the persisted-mode and
    source-identity bookkeeping must be exactly as it was.
    """
    from xrd_tools.core import FrameRecord
    from xdart.modules.display_context import CommitGate

    store, original = _record_store_with_evicted_frame()
    gate = CommitGate()
    epoch = gate.epoch
    persisted_before = store.persisted_modes(1)
    source_before = store.source_identity(1)

    def latched_hydrator(label):
        # The context is retired WHILE the read is in flight — the exact race
        # §10.2 closes, seen from the store's side.
        gate.cancel()
        return FrameRecord.from_view(original)

    store.set_hydrator(latched_hydrator)
    store.get_or_hydrate(1, commit_gate=gate, commit_epoch=epoch)

    assert not store.has_heavy_payload(1), (
        "a cancelled gate still inserted the read payload")
    assert store.persisted_modes(1) == persisted_before
    assert store.source_identity(1) == source_before


# --------------------------------------------------------------------------- #
# §10.3 — completion admission is source- and epoch-qualified
# --------------------------------------------------------------------------- #

def test_completion_admission_refuses_a_foreign_source_and_a_stale_epoch(
        widget, monkeypatch, tmp_path):
    """§10.3.3/10.3.5 — the whole owner qualifies, not part of it.

    Two sub-scans of one container share a context token, a scan key, a label
    and a display generation; only the SOURCE tells them apart.  And a request
    minted before a rescope carries a superseded epoch.
    """
    widget._enter_run_state(origin=RUN_ORIGIN_REINTEGRATE)
    display = widget.displayframe
    context = widget._acquisition_context
    request = display._build_hydration_request(4, purpose="full")
    assert request is not None
    good = request.owner
    assert good.source and good.epoch

    assert displayFrameWidget._admit_hydration_owner(
        display, good, 4, request.generation) is True

    from xdart.modules.display_context import HydrationOwner

    foreign_source = HydrationOwner.of(
        context_token=good.context_token, scan_key=good.scan_key,
        source="/data/another_sub_scan.nxs", epoch=good.epoch)
    assert displayFrameWidget._admit_hydration_owner(
        display, foreign_source, 4, request.generation) is False, (
        "a completion from a different source was admitted")

    stale_epoch = HydrationOwner.of(
        context_token=good.context_token, scan_key=good.scan_key,
        source=good.source, epoch=good.epoch + 1)
    assert displayFrameWidget._admit_hydration_owner(
        display, stale_epoch, 4, request.generation) is False, (
        "a completion from a superseded epoch was admitted")
