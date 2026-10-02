"""The shared latest-request owner (``widgets/latest_request.py``; item 6's two loops made one): a tool's computations
on a ``SingleWorkerOwner`` driven by a poll timer, the latest request winning. The 3D Viewer's ``VolumeView`` and
RSM's Custom Slice window mount it; their own rows are in test_volume_view.py and test_rsm_page.py."""
from __future__ import annotations

import gc
import threading
import time
import weakref

import pytest

pytest.importorskip("pyqtgraph")

from pyqtgraph.Qt import QtWidgets

from xdart.gui.pages.operation_owner import OperationCancelled, OperationTerminalStatus
from xdart.gui.pages.values import PageCleanup


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _Result:
    """A job's result (weakly referenceable, as a tool's plane or RGBA is let go)."""

    def __init__(self, request):
        self.request = request


class _Tool:
    """A tool as VolumeView and Custom Slice mount the owner: its job (held before it returns until ``release`` is
    set; it can honour its cancel event, or fail) and ``done``, recording what it is handed."""

    def __init__(self, *, fail=False, honours_cancel=False):
        from xdart.gui.widgets.latest_request import LatestRequestOwner
        self.calls, self.results, self.handed = [], [], []
        self.computed, self.release = threading.Event(), threading.Event()
        self.release.set()
        self.fail, self.honours_cancel = fail, honours_cancel
        self.owner = LatestRequestOwner(self.job, self.done, same=lambda a, b: a == b)

    def job(self, request, cancel, _progress):
        self.calls.append(request)
        if self.fail:
            raise MemoryError("no room")
        result = _Result(request)
        self.results.append(weakref.ref(result))
        self.computed.set()
        assert self.release.wait(20)
        if self.honours_cancel and cancel.is_set():
            raise OperationCancelled()
        return result

    def done(self, request, terminal):
        payload = None if terminal.payload is None else terminal.payload.request
        self.handed.append((request, terminal.status, payload, terminal.failure_type))
        return terminal.status is OperationTerminalStatus.RETURNED

    def hold(self):
        self.release.clear()
        self.computed.clear()


def _settled(owner, seconds=20.0):
    deadline = time.monotonic() + seconds
    while (owner.computing or owner.running is not None) and time.monotonic() < deadline:
        QtWidgets.QApplication.processEvents()
        time.sleep(0.002)
    assert not owner.computing and owner.running is None
    return owner


def test_the_latest_request_wins_and_a_superseded_result_is_dropped_and_let_go(qapp):
    tool = _Tool()
    tool.hold()
    assert tool.owner.ask("A") and tool.computed.wait(20)
    assert tool.owner.ask("B")  # asked for while A is on the worker: at most one runs
    assert tool.owner.running[1] == "A" and tool.owner.wanted == "B" and tool.owner.computing
    tool.release.set()
    _settled(tool.owner)
    assert tool.calls == ["A", "B"]
    assert tool.handed == [("B", OperationTerminalStatus.RETURNED, "B", "")] and tool.owner.shown == "B"
    gc.collect()
    assert tool.results[0]() is None  # A's result, dropped
    tool.owner.shutdown()


def test_an_unchanged_request_computes_nothing_until_what_is_shown_is_forgotten(qapp):
    tool = _Tool()
    assert tool.owner.ask("A")
    _settled(tool.owner)
    assert tool.owner.ask("A") is False and not tool.owner.computing and tool.calls == ["A"]
    tool.owner.forget()  # the tool shows something else (a refusal, a cleared view)
    assert tool.owner.ask("A")
    _settled(tool.owner)
    assert tool.calls == ["A", "A"] and len(tool.handed) == 2
    tool.owner.shutdown()


def test_releasing_drops_what_is_computing_and_lets_it_go(qapp):
    tool = _Tool()
    tool.hold()
    assert tool.owner.ask("A") and tool.computed.wait(20)
    tool.owner.release()  # the tool closing
    assert not tool.owner.computing and tool.owner.shown is None
    tool.release.set()
    _settled(tool.owner)
    assert tool.handed == []
    gc.collect()
    assert tool.results[0]() is None
    assert tool.owner.ask("A")  # shown again: computed again
    _settled(tool.owner)
    assert [request for request, *_rest in tool.handed] == ["A"]
    tool.owner.shutdown()


def test_shutdown_is_pending_while_a_job_runs_then_clean_and_hands_nothing_over(qapp):
    tool = _Tool()
    tool.hold()
    assert tool.owner.ask("A") and tool.computed.wait(20)
    assert tool.owner.shutdown().status is PageCleanup.PENDING
    tool.release.set()
    deadline = time.monotonic() + 20
    receipt = tool.owner.shutdown()
    while receipt.status is PageCleanup.PENDING and time.monotonic() < deadline:
        time.sleep(0.01)
        receipt = tool.owner.shutdown()
    assert receipt.status is PageCleanup.CLEAN and tool.owner.running is None and not tool.owner.computing
    QtWidgets.QApplication.processEvents()
    assert tool.handed == []
    gc.collect()
    assert tool.results[0]() is None


def test_a_failed_job_is_handed_to_done_and_nothing_is_shown(qapp):
    tool = _Tool(fail=True)
    assert tool.owner.ask("A")
    _settled(tool.owner)
    assert tool.handed == [("A", OperationTerminalStatus.FAILED, None, "MemoryError")]
    assert tool.owner.shown is None
    tool.owner.shutdown()


def test_a_cancelled_request_asked_for_again_is_computed_again(qapp):
    """A, then B (A cancelled), then A again before A returns: A's cancelled run is not handed over, and A is
    computed again; B never runs."""
    tool = _Tool(honours_cancel=True)
    tool.hold()
    assert tool.owner.ask("A") and tool.computed.wait(20)
    assert tool.owner.ask("B") and tool.owner.ask("A")
    tool.release.set()
    _settled(tool.owner)
    assert tool.calls == ["A", "A"]
    assert tool.handed == [("A", OperationTerminalStatus.RETURNED, "A", "")]
    tool.owner.shutdown()
