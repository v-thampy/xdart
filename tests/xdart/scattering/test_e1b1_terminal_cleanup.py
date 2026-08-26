from __future__ import annotations

from pathlib import Path
from threading import Event, Thread
import time
from types import SimpleNamespace

import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets
import tifffile

from xrd_tools.session import Light1DCustodyState, Light1DLeaseState
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
from xdart.gui.tabs.scattering.display_runtime import RunDisplayState
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


def test_wrong_shaped_mask_fails_cleanly_and_next_run_can_start(
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real zero-frame reduction failure settles below the Qt timer."""

    raw = tmp_path / "raw_0001.tif"
    tifffile.imwrite(
        raw,
        np.arange(195 * 487, dtype=np.uint16).reshape(195, 487),
        photometric="minisblack",
    )
    poni = tmp_path / "calibration.poni"
    poni.write_text(
        "poni_version: 2\n"
        "Detector: Pilatus100k\n"
        "Detector_config: {}\n"
        "Distance: 0.1234\n"
        "Poni1: 0.01\n"
        "Poni2: 0.01\n"
        "Rot1: 0.0\n"
        "Rot2: 0.0\n"
        "Rot3: 0.0\n"
        "Wavelength: 1.0e-10\n",
        encoding="utf-8",
    )
    wrong_mask = tmp_path / "wrong-mask.npy"
    np.save(wrong_mask, np.zeros((2, 2), dtype=bool))

    lifecycle = ScatteringCoordinator()
    executor = StandardRunExecutor(join_timeout=2.0)
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent(
            source_spec=image_series_spec(raw),
            poni_file=str(poni),
            mask_file=str(wrong_mask),
            project_root=str(tmp_path),
            save_path=str(tmp_path / "failed.nxs"),
            output_mode="Overwrite",
            processing_mode="Int 2D",
            max_cores=1,
            bai_1d_args={"npt": 16},
            bai_2d_args={"npt_rad": 16, "npt_azim": 8},
        )),
        lifecycle=lifecycle,
        sources=FilesystemSourceAdapter(),
        executor=executor,
    )
    custody: dict[str, object] = {}
    real_stage = RunDisplayState.stage_light_1d

    def observe_stage(self, owner, lease, *, hooks=None, slot=None):
        custody["owner"] = owner
        custody["lease"] = lease
        if slot is not None:
            custody["slot"] = slot
        return real_stage(self, owner, lease, hooks=hooks, slot=slot)

    monkeypatch.setattr(RunDisplayState, "stage_light_1d", observe_stage)
    observed_events: list[StandardRunEvent] = []
    real_drain = executor.drain_events

    def observe_events():
        events = real_drain()
        observed_events.extend(events)
        return events

    monkeypatch.setattr(executor, "drain_events", observe_events)

    def wait_for(predicate, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            qapp.processEvents()
            if predicate():
                return True
            time.sleep(0.01)
        return False

    try:
        shell = page.findChild(ScatteringWorkspaceShell)
        assert shell is not None
        shell.run_controls.startButton.click()
        assert wait_for(lambda: any(
            event.kind is StandardEventKind.FAILED
            for event in observed_events
        ))

        terminal = next(
            event for event in observed_events
            if event.kind is StandardEventKind.FAILED
        )
        assert terminal.cleanup_status is CleanupStatus.CLEANED
        assert terminal.completed == 0
        assert terminal.primary is not None
        assert "mask shape" in terminal.primary.message.casefold()
        assert lifecycle.phase is RunPhase.FAILED
        assert lifecycle.reset_permitted
        assert shell.run_controls.startButton.isEnabled()

        owner = custody["owner"]
        lease = custody["lease"]
        slot = custody["slot"]
        assert owner.light_lease is None
        assert owner.light_slot is None
        assert lease.state is Light1DLeaseState.RELEASED
        assert slot.state is Light1DCustodyState.CANCELLED
        assert lease.authority.snapshot().reservation_count == 0
        identity = terminal.run_identity
        assert page._context_controller.run_identity is identity
        retirement_results: list[bool] = []
        real_apply_retirement = (
            page._context_controller.apply_display_retirement
        )

        def observe_retirement(receipt):
            result = real_apply_retirement(receipt)
            retirement_results.append(result)
            return result

        monkeypatch.setattr(
            page._context_controller,
            "apply_display_retirement",
            observe_retirement,
        )

        snapshot = page._intents.snapshot()
        candidate = snapshot.thaw()
        candidate.mask_file = ""
        candidate.save_path = str(tmp_path / "valid.nxs")
        accepted = page._intents.commit(
            candidate, expected_revision=snapshot.revision,
        )
        page._reconcile_snapshot(snapshot, accepted.snapshot)
        assert shell.run_controls.startButton.isEnabled()
        shell.run_controls.startButton.click()
        started = wait_for(lambda: any(
            event.kind is StandardEventKind.DISCOVERY
            and event.run_identity is not identity
            for event in observed_events
        ))
        assert started, (
            lifecycle.phase,
            lifecycle.reset_permitted,
            page._notice_text,
            page._admission_state,
            tuple(
                (event.kind, event.detail, event.cleanup_status)
                for event in observed_events
            ),
        )
        assert retirement_results == [True]
        assert page._context_controller.run_identity is not identity
    finally:
        page.close_workspace()
        page.close()
        page.deleteLater()
        qapp.processEvents()
