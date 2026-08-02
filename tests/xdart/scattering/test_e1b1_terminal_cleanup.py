from __future__ import annotations

from pathlib import Path
from threading import Event, Thread
import time
from types import SimpleNamespace

import pytest
from pyqtgraph.Qt import QtWidgets

from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import image_series_spec
from xdart.gui.tabs.scattering.adapters import run_executor as executor_module
from xdart.gui.tabs.scattering.adapters.run_executor import (
    StandardRunExecutor,
    _StandardRun,
)
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.contracts import SourceCapture
from xdart.gui.tabs.scattering.display_values import StandardEventKind, StandardRunEvent
from xdart.gui.tabs.scattering.events import (
    CleanupStatus,
    ExecutorAccepted,
    ExecutorStartFailed,
    RequestId,
    RunIdentity,
)
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xdart.gui.tabs.scattering.workspace_shell import ScatteringWorkspaceShell
from tests.xdart.scattering._admission import ImmediateAdmission, install_admission


class _OneFrameScan:
    name = "Standard"
    frames = (SimpleNamespace(index=1),)

    def __len__(self) -> int:
        return 1


class _SuccessfulSession:
    frames_completed = 0

    def start(self) -> None:
        return None

    def submit(self, _frame) -> bool:
        return False

    def finish(self, **_kwargs):
        return SimpleNamespace(
            failed=False,
            cancelled=False,
            n_processed=0,
        )

    def stop(self) -> None:
        return None


def _identity() -> RunIdentity:
    return RunIdentity(1, "f" * 64)


def _run(source: object) -> _StandardRun:
    return _StandardRun(
        None,
        _identity(),
        _OneFrameScan(),
        source,
        _SuccessfulSession(),
        None,
        Path("out.nxs"),
    )


def _terminal(executor: StandardRunExecutor) -> StandardRunEvent:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        events = executor.drain_events()
        for event in events:
            if event.kind in {
                StandardEventKind.FINISHED,
                StandardEventKind.STOPPED,
                StandardEventKind.FAILED,
            }:
                return event
        time.sleep(0.01)
    raise AssertionError("executor did not publish a terminal receipt")


def test_terminal_event_is_not_visible_before_source_cleanup_completes() -> None:
    entered = Event()
    release = Event()

    class BlockingSource:
        def close(self) -> None:
            entered.set()
            assert release.wait(5)

    executor = StandardRunExecutor()
    run = _run(BlockingSource())
    executor._active = run
    worker = Thread(target=executor._run, args=(run,))
    run.worker = worker
    worker.start()
    assert entered.wait(5)
    try:
        premature = executor.drain_events()
    finally:
        release.set()
        worker.join(5)

    assert premature == ()
    terminal = executor.drain_events()
    assert [event.kind for event in terminal] == [StandardEventKind.FINISHED]


def test_cleanup_failure_cannot_publish_false_finished() -> None:
    class FailingSource:
        def close(self) -> None:
            raise RuntimeError("source close failed")

    executor = StandardRunExecutor()
    run = _run(FailingSource())
    executor._active = run

    executor._run(run)

    terminal = executor.drain_events()
    assert [event.kind for event in terminal] == [StandardEventKind.FAILED]
    assert "source close failed" in terminal[0].detail


def test_construct_cleanup_failure_is_not_reported_cleaned(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_spec = image_series_spec(tmp_path / "raw_0001.tif")
    configuration = RunIntent(
        source_spec=source_spec,
        poni_file=str(tmp_path / "calibration.poni"),
        save_path=str(tmp_path / "output.nxs"),
        output_mode="Overwrite",
    ).freeze()
    identity = RunIdentity.from_configuration(configuration)

    class FailingSource:
        close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            raise RuntimeError("source close failed")

    source = FailingSource()
    monkeypatch.setattr(executor_module, "open_source", lambda _spec: source)
    monkeypatch.setattr(
        executor_module,
        "load_poni",
        lambda _path: (_ for _ in ()).throw(RuntimeError("PONI load failed")),
    )

    executor = StandardRunExecutor()
    capture = SourceCapture(RequestId(1), 1, source_spec)
    admission = install_admission(executor, configuration, capture)
    result = executor.start(
        configuration,
        capture,
        identity,
        admission,
    )

    assert type(result) is ExecutorAccepted
    terminal = _terminal(executor)
    assert terminal.kind is StandardEventKind.FAILED
    assert terminal.cleanup_status is CleanupStatus.CLEANUP_PENDING
    assert source.close_calls == 1


@pytest.fixture
def qapp() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_page_does_not_acknowledge_failed_executor_close(
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
) -> None:
    class FailingCloseExecutor(ImmediateAdmission):
        close_calls = 0

        def start(self, _configuration, _source, run_identity, _admission):
            return ExecutorAccepted(run_identity)

        def stop(self, _run_identity) -> None:
            return None

        def close(self, _run_identity) -> None:
            self.close_calls += 1
            raise RuntimeError("executor remains open")

        def pause(self, _run_identity) -> None:
            return None

        def resume(self, _run_identity) -> None:
            return None

        def drain_events(self):
            return ()

    source = image_series_spec(tmp_path / "raw_0001.tif")
    lifecycle = ScatteringCoordinator()
    executor = FailingCloseExecutor()
    page = ScatteringWorkspace(
        intents=RunIntentStore(
            RunIntent(
                    source_spec=source,
                    poni_file=str(tmp_path / "calibration.poni"),
                    save_path=str(tmp_path / "output.nxs"),
                    output_mode="Overwrite",
            )
        ),
        lifecycle=lifecycle,
        sources=FilesystemSourceAdapter(),
        executor=executor,
    )
    try:
        shell = page.findChild(ScatteringWorkspaceShell)
        assert shell is not None
        shell.commandRequested.emit(
            ShellCommand(ShellCommandKind.RUN_ACTION)
        )
        page._drain_executor()
        qapp.processEvents()
        assert lifecycle.phase is RunPhase.RUNNING

        page.close_workspace()

        assert executor.close_calls == 1
        assert lifecycle.phase is RunPhase.STOPPING
    finally:
        page.close()
        page.deleteLater()
        qapp.processEvents()
