from __future__ import annotations

import time
from pathlib import Path
from threading import Event, Thread

from pyqtgraph.Qt import QtWidgets

from tests.xdart.scattering._e2sd_support import Sources, directory_start
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.contracts import (
    AdmissionFailure,
    AdmissionReceipt,
    AdmissionToken,
)
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.events import CleanupStatus
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.shell_values import (
    ShellCommand,
    ShellCommandKind,
)
from xdart.gui.tabs.scattering.start_outcomes import StartRefusal
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xdart.gui.tabs.scattering.workspace_shell import ScatteringWorkspaceShell
from xdart.gui.tabs.scattering.adapters import run_executor as executor_module


def _shell(page: ScatteringWorkspace) -> ScatteringWorkspaceShell:
    shell = page.findChild(ScatteringWorkspaceShell)
    assert shell is not None
    return shell


def _run(page: ScatteringWorkspace) -> None:
    _shell(page).commandRequested.emit(
        ShellCommand(ShellCommandKind.RUN_ACTION)
    )


def test_real_page_failure_after_output_inspection_does_not_deadlock_run(
    tmp_path: Path, monkeypatch,
) -> None:
    run_store, _capture = directory_start(tmp_path)
    from xdart.gui.tabs.scattering.adapters import run_executor as executor_module
    from xdart.gui.tabs.scattering import output_preflight

    monkeypatch.setattr(
        output_preflight,
        "inspect_output",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("post-reservation inspection failed")
        ),
    )
    monkeypatch.setattr(
        executor_module, "build_admission_receipt", output_preflight.prepare_output
    )
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    lifecycle = ScatteringCoordinator()
    executor = StandardRunExecutor()
    page = ScatteringWorkspace(
        intents=run_store,
        lifecycle=lifecycle,
        sources=Sources(),
        executor=executor,
    )
    try:
        _run(page)
        deadline = time.monotonic() + 10
        while lifecycle.phase is not RunPhase.IDLE and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        # Allow the page's failure branch to finish after lifecycle refusal.
        for _ in range(10):
            app.processEvents()
            time.sleep(0.005)

        assert lifecycle.phase is RunPhase.IDLE
        assert page._admission is None
        assert executor._admission is None
        assert _shell(page).run_controls.startButton.isEnabled()
    finally:
        page.close_workspace()


def test_late_session_registration_rearms_terminal_cleanup(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cleanup_entered = Event()
    allow_cleanup = Event()

    class LateSession:
        calls = 0

        def close(self) -> None:
            self.calls += 1

    _run_store, start = directory_start(tmp_path)
    operation = executor_module._AdmissionOperation(
        AdmissionToken(start.request_id, start.intent_snapshot.revision), start
    )
    operation.request_cancel()
    executor = StandardRunExecutor()
    executor._admission = operation
    original_cleanup_once = type(operation).cleanup_once
    cleanup_calls = 0

    def block_first_cleanup(candidate):
        nonlocal cleanup_calls
        cleanup_calls += 1
        if candidate is operation and cleanup_calls == 1:
            cleanup_entered.set()
            assert allow_cleanup.wait(5)
            return True
        return original_cleanup_once(candidate)

    monkeypatch.setattr(type(operation), "cleanup_once", block_first_cleanup)
    releases = []
    release = Thread(
        target=lambda: releases.append(
            executor.release_admission(operation.token)
        )
    )
    release.start()
    assert cleanup_entered.wait(5)

    late = LateSession()
    operation.register_directory_session(late)
    assert operation.cleanup_requested
    allow_cleanup.set()
    release.join(5)

    assert not release.is_alive()
    assert cleanup_calls == 2
    assert late.calls == 1
    assert operation.directory_session is None
    assert releases[0].cleanup_status is CleanupStatus.CLEANUP_PENDING
    assert executor._admission is operation

    operation.finish_worker(
        AdmissionFailure(operation.token, "worker stopped")
    )
    executor._cleanup_admission(operation)

    assert operation.cleanup_receipt().cleanup_status is CleanupStatus.CLEANED
    assert executor._admission is None


def test_real_intent_edit_after_directory_admission_does_not_orphan_cleanup(
    tmp_path: Path,
) -> None:
    run_store, _capture = directory_start(tmp_path)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    lifecycle = ScatteringCoordinator()
    executor = StandardRunExecutor()
    page = ScatteringWorkspace(
        intents=run_store,
        lifecycle=lifecycle,
        sources=Sources(),
        executor=executor,
    )
    try:
        _run(page)
        token = page._admission
        assert token is not None
        deadline = time.monotonic() + 10
        result = None
        while result is None and time.monotonic() < deadline:
            result = executor.poll_admission(token)
            time.sleep(0.005)
        assert type(result) is AdmissionReceipt
        prior = run_store.snapshot()
        candidate = prior.thaw()
        candidate.max_cores += 1
        accepted = run_store.commit(candidate, expected_revision=prior.revision)
        page._reconcile_snapshot(prior, accepted.snapshot)

        assert page._admission is None
        assert executor._admission is None
        # The refused preparation is terminal, and the next Run can admit.
        assert lifecycle.phase is RunPhase.IDLE
        _run(page)
        assert page._admission is not None
    finally:
        page.close_workspace()


def test_reconcile_retains_pending_token_and_disables_run(
    tmp_path: Path,
) -> None:
    run_store, _capture = directory_start(tmp_path)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    lifecycle = ScatteringCoordinator()
    executor = StandardRunExecutor()
    page = ScatteringWorkspace(
        intents=run_store,
        lifecycle=lifecycle,
        sources=Sources(),
        executor=executor,
    )
    try:
        _run(page)
        token = page._admission
        assert token is not None
        deadline = time.monotonic() + 10
        result = None
        while result is None and time.monotonic() < deadline:
            result = executor.poll_admission(token)
            app.processEvents()
            time.sleep(0.005)
        assert type(result) is AdmissionReceipt
        operation = executor._admission
        assert operation is not None
        owned_session = operation.directory_session
        assert owned_session is not None

        class PersistentSession:
            failing = True

            def close(self) -> None:
                if self.failing:
                    raise RuntimeError("persistent close failure")
                owned_session.close()

        session = PersistentSession()
        operation.directory_session = session
        prior = run_store.snapshot()
        candidate = prior.thaw()
        candidate.max_cores += 1
        accepted = run_store.commit(candidate, expected_revision=prior.revision)
        page._reconcile_snapshot(prior, accepted.snapshot)

        assert page._admission is token
        assert executor._admission is operation
        assert not _shell(page).run_controls.startButton.isEnabled()
        assert (
            _shell(page).run_controls.readinessLabel.text()
            == "Output cleanup remains pending"
        )
        assert lifecycle.phase is RunPhase.IDLE

        session.failing = False
        assert (
            executor.release_admission(token).cleanup_status
            is CleanupStatus.CLEANED
        )
    finally:
        page.close_workspace()


def test_close_retains_persistent_cleanup_for_exact_retry(
    tmp_path: Path, monkeypatch,
) -> None:
    run_store, _capture = directory_start(tmp_path)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    lifecycle = ScatteringCoordinator()
    executor = StandardRunExecutor()
    page = ScatteringWorkspace(
        intents=run_store,
        lifecycle=lifecycle,
        sources=Sources(),
        executor=executor,
    )
    _run(page)
    page._run_timer.stop()
    token = page._admission
    assert token is not None
    deadline = time.monotonic() + 10
    result = None
    while result is None and time.monotonic() < deadline:
        result = executor.poll_admission(token)
        app.processEvents()
        time.sleep(0.005)
    assert type(result) is AdmissionReceipt
    operation = executor._admission
    assert operation is not None
    owned_session = operation.directory_session
    assert owned_session is not None
    class PersistentSession:
        failing = True

        def close(self) -> None:
            if self.failing:
                raise RuntimeError("persistent close failure")
            owned_session.close()

    session = PersistentSession()
    operation.directory_session = session
    calls = 0
    close_calls = 0
    original_refuse = type(page._pipeline).refuse
    original_close = type(page._pipeline).close

    def counted_refuse(pipeline, capture, reason):
        nonlocal calls
        calls += 1
        assert reason is StartRefusal.OUTPUT_PREFLIGHT
        return original_refuse(pipeline, capture, reason)

    def counted_close(pipeline):
        nonlocal close_calls
        close_calls += 1
        return original_close(pipeline)

    monkeypatch.setattr(type(page._pipeline), "refuse", counted_refuse)
    monkeypatch.setattr(type(page._pipeline), "close", counted_close)
    page.close_workspace()

    assert calls == 1
    assert close_calls == 1
    assert lifecycle.phase is RunPhase.CLOSED
    assert page._admission is token
    assert executor._admission is operation
    assert operation.cleanup_receipt().cleanup_status is CleanupStatus.CLEANUP_PENDING

    session.failing = False
    page.close_workspace()

    assert calls == 1
    assert close_calls == 1
    assert page._admission is None
    assert executor._admission is None
