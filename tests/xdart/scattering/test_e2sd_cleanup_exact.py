from __future__ import annotations

from pathlib import Path
from threading import Event, Thread
import time
from unittest.mock import patch

import pytest
from pyqtgraph.Qt import QtWidgets

from tests.xdart.scattering._e2sd_support import Sources, store
from xdart.gui.tabs.scattering.adapters import run_executor as executor_module
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.contracts import SourceCapture, StartCapture
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.events import RequestId
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.shell_values import (
    ShellCommand,
    ShellCommandKind,
)
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xdart.gui.tabs.scattering.workspace_shell import ScatteringWorkspaceShell


@pytest.fixture
def qapp() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _settle(app: QtWidgets.QApplication, predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("Qt operation did not settle")


def _shell(page: ScatteringWorkspace) -> ScatteringWorkspaceShell:
    shell = page.findChild(ScatteringWorkspaceShell)
    assert shell is not None
    return shell


def _run(page: ScatteringWorkspace) -> None:
    _shell(page).commandRequested.emit(
        ShellCommand(ShellCommandKind.RUN_ACTION)
    )


def test_append_refusal_returns_idle_then_overwrite_can_admit(
    qapp: QtWidgets.QApplication, tmp_path: Path
) -> None:
    lifecycle = ScatteringCoordinator()
    executor = StandardRunExecutor()
    run_store = store(tmp_path, output_mode="Append")
    snapshot = run_store.snapshot()
    intent = snapshot.thaw()
    intent.processing_mode = "Int 1D (XYE)"
    run_store.commit(intent, expected_revision=snapshot.revision)
    page = ScatteringWorkspace(
        intents=run_store,
        lifecycle=lifecycle,
        sources=Sources(),
        executor=executor,
    )
    try:
        shell = _shell(page)
        _run(page)
        _settle(qapp, lambda: page._admission is None)
        assert lifecycle.phase is RunPhase.IDLE
        assert executor._admission is None
        assert shell.run_controls.readinessLabel.text() == (
            "XYE-only Append has no persisted lineage owner"
        )

        shell.run_controls.writeModeButton.click()
        assert run_store.snapshot().thaw().output_mode == "Overwrite"
        _run(page)
        assert page._admission is not None
    finally:
        page.close_workspace()


def test_late_fail_once_admission_owner_is_not_lost_after_cancel(
    tmp_path: Path,
) -> None:
    entered = Event()
    allow_owner = Event()

    class FailOnceSession:
        def __init__(self) -> None:
            self.calls = 0

        def close(self) -> None:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient late close failure")

    owner = FailOnceSession()

    def late_failure(_capture, *, session_owner, **_kwargs):
        entered.set()
        assert allow_owner.wait(5)
        session_owner(owner)
        raise RuntimeError("cancelled admission finished late")

    executor = StandardRunExecutor()
    run_store = store(tmp_path)
    source = run_store.snapshot().thaw().source_spec
    request = RequestId(1)
    capture = SourceCapture(request, 1, source)
    start = StartCapture(request, 1, run_store.snapshot(), capture)

    with patch.object(
        executor_module, "build_admission_receipt", side_effect=late_failure
    ):
        token = executor.begin_admission(start)
        assert entered.wait(5)
        released = executor.release_admission(token)
        assert released.cleanup_status.value != "cleanup_failed"
        allow_owner.set()
        deadline = time.monotonic() + 2.0
        while owner.calls < 2 and time.monotonic() < deadline:
            time.sleep(0.005)

    assert owner.calls == 2
    assert executor._admission is None


def test_late_persistent_owner_remains_reachable_after_cancel(
    tmp_path: Path,
) -> None:
    entered = Event()
    allow_owner = Event()

    class PersistentFailure:
        def __init__(self) -> None:
            self.calls = 0

        def close(self) -> None:
            self.calls += 1
            raise RuntimeError("persistent late close failure")

    owner = PersistentFailure()

    def late_failure(_capture, *, session_owner, **_kwargs):
        entered.set()
        assert allow_owner.wait(5)
        session_owner(owner)
        raise RuntimeError("admission finished late")

    executor = StandardRunExecutor()
    run_store = store(tmp_path)
    source = run_store.snapshot().thaw().source_spec
    request = RequestId(1)
    capture = SourceCapture(request, 1, source)
    start = StartCapture(request, 1, run_store.snapshot(), capture)

    with patch.object(
        executor_module, "build_admission_receipt", side_effect=late_failure
    ):
        token = executor.begin_admission(start)
        operation = executor._admission
        assert operation is not None
        assert entered.wait(5)
        released = executor.release_admission(token)
        assert released.cleanup_status.value == "cleanup_pending"
        allow_owner.set()
        deadline = time.monotonic() + 2.0
        while owner.calls < 2 and time.monotonic() < deadline:
            time.sleep(0.005)

    assert owner.calls == 2
    assert executor._admission is operation
    assert operation.directory_session is owner


def test_concurrent_duplicate_release_cannot_forget_pending_owner(
    tmp_path: Path,
) -> None:
    entered = Event()
    release_first = Event()

    class BlockingFailure:
        def __init__(self) -> None:
            self.calls = 0

        def close(self) -> None:
            self.calls += 1
            if self.calls == 1:
                entered.set()
                assert release_first.wait(5)
            raise RuntimeError("owner is still live")

    owner = BlockingFailure()
    executor = StandardRunExecutor()
    run_store = store(tmp_path)
    source = run_store.snapshot().thaw().source_spec
    request = RequestId(1)
    capture = SourceCapture(request, 1, source)
    start = StartCapture(request, 1, run_store.snapshot(), capture)
    operation = executor_module._AdmissionOperation(
        executor_module.AdmissionToken(request, 0), start
    )
    operation.register_directory_session(owner)
    executor._admission = operation

    first = Thread(target=executor.release_admission, args=(operation.token,))
    first.start()
    assert entered.wait(5)
    second = executor.release_admission(operation.token)
    assert second.cleanup_status.value == "cleanup_pending"
    release_first.set()
    first.join(5)

    assert owner.calls == 2
    assert executor._admission is operation
    assert operation.directory_session is owner
