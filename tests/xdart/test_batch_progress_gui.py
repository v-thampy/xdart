"""NEXT-128-WINDOW-POLISH item 14, Batch progress (the maintainer's live check, 2026-11-03: "i started a batch fit over
a long sequence. there is no update progress and i have no idea where things stand"; his "Count + live trend
(Recommended)"). The shared latest-request owner hands a running request's progress to its tool (never a released
one's); the fit host counts a Batch's frames in the window's status and streams each fitted frame into the trend; the
trend takes many rows in one redraw, so a long batch's end no longer redraws once per frame (measured: 3621 rows,
26.91 s through ``set_row`` against one 11.2 ms redraw)."""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

pytest.importorskip("pyqtgraph")

from pyqtgraph.Qt import QtWidgets


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _until(predicate, seconds=20.0):
    deadline = time.monotonic() + seconds
    while not predicate() and time.monotonic() < deadline:
        QtWidgets.QApplication.processEvents()
        time.sleep(0.005)
    return predicate()


def _sample(identity="scan"):
    from xrd_tools.analysis.fit_snapshot import FitAxis, FitInputKey, FitInputSnapshot
    x = np.linspace(1., 4., 301)
    y = 20 + 500 * np.exp(-.5 * ((x - 2.5) / .03) ** 2)
    return FitInputSnapshot(FitInputKey(identity, "/data/" + identity + ".nexus", "admitted", 1),
                            (FitAxis("q", "q_A^-1", x),), y, wavelength_angstrom=1.)


def test_the_owner_hands_a_wanted_requests_progress_to_its_tool_and_none_once_released(qapp):
    from xdart.gui.widgets.latest_request import LatestRequestOwner
    gate, seen, handed = threading.Event(), [], []

    def job(request, cancel, progress):
        if request == "released":
            assert gate.wait(20)
            progress("Fitting", 1, 3)
            return request
        progress("Fitting", 1, 3)
        assert gate.wait(20)
        gate.clear()
        progress("Fitting", 2, 3)
        assert gate.wait(20)
        gate.clear()
        return request

    owner = LatestRequestOwner(job, lambda request, terminal: handed.append(request) or True,
                               same=lambda a, b: a is b,
                               progress=lambda request, update: seen.append(
                                   (request, update.stage, update.completed, update.total)))
    owner.ask("wanted")
    assert _until(lambda: seen == [("wanted", "Fitting", 1, 3)]), seen
    gate.set()
    assert _until(lambda: len(seen) == 2), seen
    assert seen[1] == ("wanted", "Fitting", 2, 3)
    gate.set()
    assert _until(lambda: handed == ["wanted"]), handed
    assert not owner.computing
    # a released request's progress reaches nobody, and its result is dropped
    owner.ask("released")
    owner.release()
    gate.set()
    assert _until(lambda: owner.running is None)
    for _ in range(20):
        QtWidgets.QApplication.processEvents()
        time.sleep(0.005)
    assert seen == [("wanted", "Fitting", 1, 3), ("wanted", "Fitting", 2, 3)] and handed == ["wanted"]
    while owner.shutdown().status.value == "pending":
        QtWidgets.QApplication.processEvents()


def test_the_trend_takes_many_rows_in_one_redraw(qapp):
    from xdart.gui.tabs.scattering.trend_view import TrendView
    rows = [(f"frame {i}", float(i), {"center_0": 2.5 + 1e-3 * i, "fwhm_0": .07}) for i in range(40)]
    one, bulk = TrendView(None, name="trendOne"), TrendView(None, name="trendBulk")
    for identity, x, params in rows:
        one.set_row(identity, x, params)
    redraws = []
    original = bulk.redraw
    bulk.redraw = lambda *args: (redraws.append(1), original(*args))
    bulk.set_rows(rows)
    assert len(redraws) == 1
    assert bulk.table()[0] == one.table()[0] and bulk.table()[1] == one.table()[1]
    assert bulk.table()[2] == one.table()[2]
    bulk.set_rows([])  # nothing to add: nothing redrawn
    assert len(redraws) == 1


def test_a_batch_counts_its_frames_and_streams_each_fitted_frame_into_the_trend(qapp, monkeypatch):
    from xdart.gui.analysis.snapshot_fit_host import FitTarget, SnapshotFitHost
    from xdart.gui.tabs.scattering.trend_view import TrendView
    gates = [threading.Event() for _ in range(3)]

    def target(index):
        def load():
            assert gates[index].wait(20)
            return _sample(f"frame{index}")
        return FitTarget(f"frame {index}", load, x=float(10 + index))

    state = [_sample()]
    host = SnapshotFitHost("peak", current=lambda: state[0], scope=lambda: state[0].key,
                           batch=lambda: tuple(target(index) for index in range(3)))
    per_row = []
    original = TrendView.set_row
    monkeypatch.setattr(TrendView, "set_row", lambda self, *args, **kwargs: (per_row.append(args[0]),
                                                                              original(self, *args, **kwargs)))
    counted = []
    count = host.dialog.set_batch_progress
    host.dialog.set_batch_progress = lambda done, total: (counted.append((done, total)), count(done, total))
    host.dialog.show()
    host.reload()
    assert _until(lambda: not host.computing)
    host.dialog.fit_btn.click()  # one Fit counts no batch
    assert _until(lambda: not host.computing) and host.outcomes
    assert counted == []
    status, trend = host.dialog.status, host.dialog.trend
    host.dialog.batch_btn.click()
    assert host.computing and trend.table()[0] == ()  # the earlier result's rows go as the batch starts
    gates[0].set()
    assert _until(lambda: status.text() == "Batch fitting… 1/3"), status.text()
    assert _until(lambda: len(trend.table()[0]) == 1), trend.table()[0]
    assert trend.table()[1] == (10.0,)
    gates[1].set()
    assert _until(lambda: status.text() == "Batch fitting… 2/3"), status.text()
    assert _until(lambda: len(trend.table()[0]) == 2), trend.table()[0]  # held rows reach the trend unprompted
    assert trend.table()[1] == (10.0, 11.0)
    gates[2].set()
    assert _until(lambda: not host.computing)
    assert len(host.outcomes) == 3 and all(outcome.ok for outcome in host.outcomes)
    assert trend.table()[1] == (10.0, 11.0, 12.0)
    assert status.text() == "Batch done: 3 of 3 frames fitted", status.text()
    assert per_row == []  # streamed and final rows go through the trend's one-redraw bulk
    assert counted[-1] == (3, 3) or counted[-1] == (2, 3)
    host.dialog.close()
    while host.shutdown().status.value == "pending":
        QtWidgets.QApplication.processEvents()


def test_a_peak_batch_is_refused_up_front_while_its_degree_is_refused(qapp):
    """The background degree (item 14): a Peak Batch checks the window's refusal before it starts, as Phase Analysis's
    does, instead of refusing every frame by the plan."""
    from xdart.gui.analysis.snapshot_fit_host import FitTarget, SnapshotFitHost
    state = [_sample()]
    host = SnapshotFitHost("peak", current=lambda: state[0], scope=lambda: state[0].key,
                           batch=lambda: tuple(FitTarget(f"frame {index}", lambda index=index: _sample(f"f{index}"))
                                               for index in range(2)))
    dialog = host.dialog
    dialog.show()
    assert dialog.bkg_degree_row.isVisible()  # Polynomial, the default, takes a degree
    dialog.bkg_degree_row.editor.setText("9")  # Polynomial takes 0 to 7
    dialog.batch_btn.click()
    assert not host.computing, "no batch starts"
    assert dialog.status.text().startswith("Degree:"), dialog.status.text()
    dialog.close()
    while host.shutdown().status.value == "pending":
        QtWidgets.QApplication.processEvents()
