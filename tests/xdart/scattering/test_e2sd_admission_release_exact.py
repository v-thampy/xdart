from __future__ import annotations

import time
from pathlib import Path
from threading import Event, Thread

from pyqtgraph.Qt import QtWidgets

from tests.xdart.scattering._e2sd_support import Sources, directory_start
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.adapters.target_reservation import TargetLease
from xdart.gui.tabs.scattering.contracts import AdmissionFailure, AdmissionReceipt
from xdart.gui.tabs.scattering.contracts import AdmissionToken, SourceCapture, StartCapture
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.events import CleanupStatus, RequestId
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


def test_one_release_closes_directory_then_releases_target(tmp_path: Path) -> None:
    _store, capture = directory_start(tmp_path)
    executor = StandardRunExecutor()
    token = executor.begin_admission(capture)
    deadline = time.monotonic() + 10
    result = None
    while result is None and time.monotonic() < deadline:
        result = executor.poll_admission(token)
        time.sleep(0.005)
    assert type(result) is AdmissionReceipt, getattr(result, "reason", result)
    operation = executor._admission
    assert operation is not None
    assert operation.directory_session is not None
    assert operation.target_lease is not None
    paths = operation.target_lease.paths

    released = executor.release_admission(token)

    assert released.cleanup_status is CleanupStatus.CLEANED
    assert executor._admission is None
    assert all(path not in TargetLease._reserved for path in paths)


def test_failed_admission_after_both_owners_does_not_need_second_release(
    tmp_path: Path, monkeypatch,
) -> None:
    _store, capture = directory_start(tmp_path)
    from xdart.gui.tabs.scattering.adapters import run_executor as executor_module
    from xdart.gui.tabs.scattering import output_preflight

    original = output_preflight.inspect_output

    def fail_after_reservation(*args, **kwargs):
        raise RuntimeError("post-reservation inspection failed")

    monkeypatch.setattr(output_preflight, "inspect_output", fail_after_reservation)
    monkeypatch.setattr(executor_module, "build_admission_receipt", output_preflight.prepare_output)
    executor = StandardRunExecutor()
    token = executor.begin_admission(capture)
    deadline = time.monotonic() + 10
    result = None
    while result is None and time.monotonic() < deadline:
        result = executor.poll_admission(token)
        time.sleep(0.005)
    assert type(result) is AdmissionFailure
    operation = executor._admission
    assert operation is not None
    assert operation.directory_session is not None
    assert operation.target_lease is not None
    paths = operation.target_lease.paths

    released = executor.release_admission(token)

    assert released.cleanup_status is CleanupStatus.CLEANED
    assert executor._admission is None
    assert all(path not in TargetLease._reserved for path in paths)
    assert original is not None


def test_real_page_failure_after_reservation_does_not_deadlock_run(
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


def test_late_target_registration_during_cleanup_is_cleaned_without_third_call(
    tmp_path: Path,
) -> None:
    close_entered = Event()
    allow_close = Event()

    class BlockingSession:
        def close(self) -> None:
            close_entered.set()
            assert allow_close.wait(5)

    run_store, start = directory_start(tmp_path)
    operation = executor_module._AdmissionOperation(
        AdmissionToken(start.request_id, start.intent_snapshot.revision), start
    )
    operation.register_directory_session(BlockingSession())
    executor = StandardRunExecutor()
    executor._admission = operation
    operation.request_cancel()

    release = Thread(target=executor.release_admission, args=(operation.token,))
    release.start()
    assert close_entered.wait(5)
    target = tmp_path / "late-target.nxs"
    try:
        operation.reserve_targets((target,))
    except RuntimeError:
        pass
    operation.finish_worker(AdmissionFailure(operation.token, "cancelled"))
    # The worker's finally-path request collides with the first cleanup.
    executor._cleanup_admission(operation)
    allow_close.set()
    release.join(5)

    assert executor._admission is None
    assert operation.cleanup_receipt().cleanup_status is CleanupStatus.CLEANED
    assert target.resolve(strict=False) not in TargetLease._reserved


def test_target_lease_remains_until_admission_worker_is_done(
    tmp_path: Path,
) -> None:
    _store, start = directory_start(tmp_path)

    class Session:
        calls = 0

        def close(self) -> None:
            self.calls += 1

    operation = executor_module._AdmissionOperation(
        AdmissionToken(start.request_id, start.intent_snapshot.revision), start
    )
    session = Session()
    operation.register_directory_session(session)
    target = tmp_path / "still-inspecting.nxs"
    operation.reserve_targets((target,))
    lease = operation.target_lease
    operation.request_cancel()
    executor = StandardRunExecutor()
    executor._admission = operation

    released = executor.release_admission(operation.token)

    assert released.cleanup_status is CleanupStatus.CLEANUP_PENDING
    assert session.calls == 1
    assert operation.directory_session is None
    assert operation.target_lease is lease
    assert target.resolve(strict=False) in TargetLease._reserved

    operation.finish_worker(
        AdmissionFailure(operation.token, "worker stopped")
    )
    executor._cleanup_admission(operation)

    assert operation.cleanup_receipt().cleanup_status is CleanupStatus.CLEANED
    assert executor._admission is None
    assert target.resolve(strict=False) not in TargetLease._reserved


def test_late_session_registration_rearms_terminal_cleanup(
    tmp_path: Path,
) -> None:
    release_entered = Event()
    allow_release = Event()

    class BlockingLease:
        def __init__(self, owned) -> None:
            self.owned = owned

        def release(self) -> None:
            release_entered.set()
            assert allow_release.wait(5)
            self.owned.release()

    class LateSession:
        calls = 0

        def close(self) -> None:
            self.calls += 1

    _run_store, start = directory_start(tmp_path)
    operation = executor_module._AdmissionOperation(
        AdmissionToken(start.request_id, start.intent_snapshot.revision), start
    )
    target = tmp_path / "initial-target.nxs"
    operation.reserve_targets((target,))
    operation.target_lease = BlockingLease(operation.target_lease)
    operation.finish_worker(AdmissionFailure(operation.token, "cancelled"))
    operation.request_cancel()
    executor = StandardRunExecutor()
    executor._admission = operation

    release = Thread(target=executor.release_admission, args=(operation.token,))
    release.start()
    assert release_entered.wait(5)
    late = LateSession()
    operation.register_directory_session(late)
    allow_release.set()
    release.join(5)

    assert late.calls == 1
    assert executor._admission is None
    assert operation.cleanup_receipt().cleanup_status is CleanupStatus.CLEANED
    assert target.resolve(strict=False) not in TargetLease._reserved


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
        operation = executor._admission
        assert operation is not None
        paths = operation.target_lease.paths

        prior = run_store.snapshot()
        candidate = prior.thaw()
        candidate.max_cores += 1
        accepted = run_store.commit(candidate, expected_revision=prior.revision)
        page._reconcile_snapshot(prior, accepted.snapshot)

        assert page._admission is None
        assert executor._admission is None
        assert all(path not in TargetLease._reserved for path in paths)
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
    paths = operation.target_lease.paths

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
    assert all(path in TargetLease._reserved for path in paths)

    session.failing = False
    page.close_workspace()

    assert calls == 1
    assert close_calls == 1
    assert page._admission is None
    assert executor._admission is None
    assert all(path not in TargetLease._reserved for path in paths)
