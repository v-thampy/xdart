"""Quarantine keeps cleanup truth, exact retry ownership and the real process deadline separate."""
from __future__ import annotations

from threading import Event
import subprocess
import sys
import os
import textwrap
import time
from types import SimpleNamespace

import pytest

from tests.xdart.scattering.test_browse_reference_rings_run import _page, _pump
from tests.xdart.scattering.test_failed_run_cleanup_page import _failed_run, _intent, older_output  # noqa: F401
from tests.xdart.scattering.test_failed_cleanup_time_limit import stuck_cleanup  # noqa: F401
from tests.xdart.scattering.test_predicted_ring_overlays import qapp  # noqa: F401
from xdart.gui.tabs.scattering import page as page_module
from xdart.gui.tabs.scattering.adapters.dynamic_output import DynamicOutputAdapter
from xdart.gui.tabs.scattering.events import CleanupStatus, ExecutorClosed, RunIdentity, RequestId
from xdart.gui.tabs.scattering.contracts import AdmissionToken
from xdart.gui.tabs.scattering.display_retirement import DisplayRetirementOwner
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.adapters.target_reservation import AdmissionOperation


def test_quarantined_proof_cannot_authorize_admission_over_another_active_run():
    old, current = RunIdentity(1, "old"), RunIdentity(2, "current")
    owner = DisplayRetirementOwner(old, lambda: pytest.fail("quarantine must not retry"))
    assert owner.quarantine(ExecutorClosed(old, CleanupStatus.CLEANUP_PENDING, quarantined=True))
    executor = StandardRunExecutor()
    executor._active = SimpleNamespace(identity=current, closed=False)
    # Capture is opaque to retirement; the real owner and admission command are exercised.
    operation = AdmissionOperation(AdmissionToken(RequestId(1), 0), None, retirement_owner=owner)
    proof = executor._retire_display(operation)
    assert not proof.released_for_admission
    assert operation.retirement_receipt is proof


def test_cancelled_admission_forgets_a_consumed_quarantined_retirement():
    identity = RunIdentity(1, "old")
    owner = DisplayRetirementOwner(identity, lambda: pytest.fail("quarantine must not retry"))
    assert owner.quarantine(ExecutorClosed(identity, CleanupStatus.CLEANUP_PENDING, quarantined=True))
    operation = AdmissionOperation(AdmissionToken(RequestId(1), 0), None, retirement_owner=owner,
                                   retirement_receipt=owner.proof, worker_done=True)
    executor = StandardRunExecutor()
    executor._retirement, executor._admission = owner, operation
    operation.request_cancel()
    executor._cleanup_admission(operation)
    assert executor._admission is None
    assert executor._retirement is None


def test_give_up_does_not_replace_an_in_flight_retirement_owner(
        qapp, tmp_path, older_output, stuck_cleanup, monkeypatch):
    with _page(qapp, _intent(tmp_path)) as (page, _view, observed):
        _failed_run(qapp, page, older_output, observed)
        page._run_timer.stop()
        identity = page._lifecycle.attempt_run_identity
        executor = page._run_executor
        run = executor._exact_run(identity)
        assert _pump(qapp, lambda: run.worker is None or not run.worker.is_alive(), 5.)
        entered, release = Event(), Event()
        retire = run.display.retire

        def held_retire(**kwargs):
            entered.set()
            assert release.wait(5.), "the test did not release the actual retirement call"
            return retire(**kwargs)

        monkeypatch.setattr(run.display, "retire", held_retire)
        try:
            executor.close(identity, background=True)
            assert _pump(qapp, entered.is_set, 5.)
            owner = executor._retirement
            assert executor.give_up(identity) is None
            assert executor._retirement is owner and executor._exact_run(identity) is run
        finally:
            monkeypatch.undo()
            release.set()
            assert _pump(qapp, lambda: executor.close(identity, background=True).cleanup_status
                         is CleanupStatus.CLEANED, 10.)


def test_a_different_cleanup_failure_restarts_the_give_up_clock(
        qapp, tmp_path, older_output, stuck_cleanup, monkeypatch):
    failing = DynamicOutputAdapter.finish_all
    message = ["cleanup failure A"]

    def failure(adapter, **kwargs):
        try:
            return failing(adapter, **kwargs)
        except RuntimeError as error:
            raise RuntimeError(message[0]) from error

    monkeypatch.setattr(DynamicOutputAdapter, "finish_all", failure)
    monkeypatch.setattr(page_module, "_FAILED_CLEANUP_RETRY_S", .05)
    monkeypatch.setattr(page_module, "_FAILED_CLEANUP_GIVE_UP_S", 4.)
    with _page(qapp, _intent(tmp_path)) as (page, _view, observed):
        _failed_run(qapp, page, older_output, observed)
        assert _pump(qapp, lambda: page._failed_cleanup_since is not None, 5.)
        started = page._failed_cleanup_since[-1]
        assert _pump(qapp, lambda: time.monotonic() - started >= 2.3, 4.)
        message[0] = "cleanup failure B"
        assert _pump(qapp, lambda: time.monotonic() - started >= 4.3, 4.)
        assert not page._lifecycle.reset_permitted, "a new failure inherited the previous failure's clock"
        assert _pump(qapp, lambda: page._lifecycle.reset_permitted, 5.)


@pytest.mark.parametrize("blocked", ["close", "logging"])
def test_real_process_close_deadline_survives_blocked_callbacks_and_logging(tmp_path, blocked):
    # Real host/process termination lives in a child; a faulty close/log boundary cannot exit pytest itself.
    code = textwrap.dedent('''
        import logging, threading
        from pyqtgraph.Qt import QtWidgets
        from xdart import _gui_main
        from tests.xdart.pages.test_synthetic_page_host import _descriptor, _services
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        _gui_main._CLOSE_PENDING_LIMIT_S = .25
        window = _gui_main.Main(page_descriptors=(_descriptor("deadline", []),), host_services=_services([]))
        window._ends_process = True
        if MODE == "close":
            real = window._close_page_and_tools
            def stuck():
                real()
                print("entered-close", flush=True)
                threading.Event().wait(30.)
                raise RuntimeError("deadline did not stop the process")
            window._close_page_and_tools = stuck
            window.close()
        else:
            handler = logging.StreamHandler()
            logging.getLogger().addHandler(handler)
            ready = threading.Event()
            def hold_log():
                handler.acquire()
                ready.set()
                threading.Event().wait(30.)
            threading.Thread(target=hold_log, daemon=True).start()
            ready.wait()
            print("entered-logging", flush=True)
            window._end_process()
        raise RuntimeError("forced exit returned")
    ''').replace("MODE", repr(blocked))
    env = {**os.environ, "XDART_SESSION_FILE": str(tmp_path / "child-session.json"),
           "XDART_SETTINGS_FILE": str(tmp_path / "child-settings.ini"),
           "XDART_PHASE_LIBRARY": str(tmp_path / "child-phases.json"), "MPLCONFIGDIR": str(tmp_path / "mpl")}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=10., env=env)
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert f"entered-{blocked}" in result.stdout, (result.stdout, result.stderr)
