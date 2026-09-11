from __future__ import annotations

from pathlib import Path
import os
from threading import Event, Thread
import time

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


def _prepared_run(tmp_path, *, frame_count=1, intent=None):
    """Build the real admission and run owners without launching the worker."""
    from tests.xdart.scattering._e2sd_support import write_poni
    from tests.xdart.scattering.test_p1b_output_graph import (
        _admit, _intent, _write_tiff,
    )
    from xdart.gui.tabs.scattering.contracts import AdmissionReceipt

    if intent is None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        raw = tmp_path / "raw_0001.tif"
        for label in range(1, frame_count + 1):
            _write_tiff(tmp_path / f"raw_{label:04d}.tif", label)
        poni = tmp_path / "cal.poni"
        write_poni(poni)
        intent = _intent(raw, tmp_path / "processed", poni)
    executor = StandardRunExecutor()
    admission, capture, _token = _admit(executor, intent, request_value=1)
    assert type(admission) is AdmissionReceipt
    configuration = intent.freeze()
    resources = executor._admission.transfer(admission)
    assert resources is not None
    executor._admission = None
    run = _StandardRun(
        configuration, RunIdentity.from_configuration(configuration),
        None, None, None, None, Path(configuration.save_path),
        capture=capture, resources=resources,
    )
    run.pending_partition_count = max(1, len(admission.outputs))
    run.total = sum(row.item.source_stamp.frame_count for row in admission.outputs)
    run.display.set_factories(executor_module.FrameRecordStore, executor_module.PublicationStore)
    run.display.bind_transport(event_sink=executor._events.put)
    executor._active = run
    return executor, run, admission


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


def test_terminal_event_is_not_visible_before_source_cleanup_completes(
    tmp_path, monkeypatch,
) -> None:
    entered = Event()
    release = Event()

    from xrd_tools.sources.image import TiffSeriesSource
    executor, run, _admission = _prepared_run(tmp_path)
    def close_source(self):
        entered.set()
        assert release.wait(5)
    monkeypatch.setattr(TiffSeriesSource, "close", close_source, raising=False)
    worker = Thread(target=executor._run, args=(run,))
    run.worker = worker
    worker.start()
    assert entered.wait(5)
    try:
        premature = executor.drain_events()
    finally:
        release.set()
        worker.join(5)

    assert not any(event.kind in {
        StandardEventKind.FINISHED, StandardEventKind.STOPPED,
        StandardEventKind.FAILED,
    } for event in premature)
    terminal = executor.drain_events()
    assert [event.kind for event in terminal] == [StandardEventKind.FINISHED]
    assert terminal[0].completed == terminal[0].total == 1
    assert executor.close(run.identity).cleanup_status is CleanupStatus.CLEANED


def test_cleanup_failure_cannot_publish_false_finished(tmp_path, monkeypatch) -> None:
    from xrd_tools.sources.image import TiffSeriesSource
    executor, run, _admission = _prepared_run(tmp_path)
    def close_source(self):
        raise RuntimeError("source close failed")
    monkeypatch.setattr(TiffSeriesSource, "close", close_source, raising=False)

    executor._run(run)

    terminal = tuple(event for event in executor.drain_events()
                     if event.kind in {StandardEventKind.FINISHED, StandardEventKind.FAILED})
    assert [event.kind for event in terminal] == [StandardEventKind.FAILED]
    assert "source close failed" in terminal[0].detail
    assert terminal[0].cleanup_status is CleanupStatus.CLEANUP_PENDING
    monkeypatch.setattr(TiffSeriesSource, "close", lambda self: None)
    assert executor.close(run.identity).cleanup_status is CleanupStatus.CLEANED


def test_construct_cleanup_failure_is_not_reported_cleaned(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog,
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
    assert any("[RUN-CLEANUP]" in row.message
               and "source.close: source close failed" in row.message
               for row in caplog.records)


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


@pytest.mark.parametrize("failure", ["mask", "refused-target", "eiger-refused-target", "eiger-nexus-only-refusal"])
def test_zero_frame_failure_cleans_up_and_next_run_can_start(
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
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
    # The production output policy adds the operation slot to the save stem.
    target = tmp_path / "failed_int2d.nexus"
    if failure != "mask":
        target.write_bytes(b"not a processed result; must not be replaced")
    before = target.read_bytes() if target.exists() else None
    real_eiger = failure.startswith("eiger-")
    if real_eiger:
        data_root = Path(os.environ.get("XDART_TEST_DATA", "/missing"))
        raw = data_root / "eiger" / "long" / "eiger_S069Ta_redo_eta2p0_1_scan001_master.h5"
        poni = data_root / "eiger" / "LaB6_detxn26_detyn6p5_eta4p5.poni"
        if not raw.is_file() or not poni.is_file():
            pytest.skip("real 651-frame Eiger input unavailable")

    lifecycle = ScatteringCoordinator()
    executor = StandardRunExecutor(join_timeout=2.0)
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent(
            source_spec=image_series_spec(raw),
            poni_file=str(poni),
            mask_file=str(wrong_mask) if failure == "mask" else "",
            project_root=str(tmp_path),
            save_path=str(tmp_path / "failed.nexus"),
            output_mode="Overwrite",
            processing_mode="Int 2D",
            max_cores=4 if real_eiger else 1,
            bai_1d_args={"npt": 1000 if real_eiger else 16},
            bai_2d_args={"npt_rad": 500 if real_eiger else 16,
                         "npt_azim": 500 if real_eiger else 8},
            run_options=({
                "_post_g2_pipeline_v2": {
                    "writer_settlement_batch_size": 1,
                    "nexus_record_batch_size": 8,
                    "reduction_inflight": 16,
                    "semantic_checkpoint_frame_cap": 56,
                    "staging_frame_cap": 64,
                },
                "_post_g2_output_diagnostics_v1": {
                    "save_xye": False, "durable_fsync": False,
                },
            } if failure == "eiger-nexus-only-refusal" else {}),
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
        )), (lifecycle.phase, page._notice_text, page._admission_state,
             tuple((event.kind, event.detail, event.cleanup_status)
                   for event in observed_events))

        terminal = next(
            event for event in observed_events
            if event.kind is StandardEventKind.FAILED
        )
        assert terminal.cleanup_status is CleanupStatus.CLEANED, terminal
        assert terminal.completed == 0
        assert terminal.primary is not None
        expected = "mask shape" if failure == "mask" else "not a current xdart processed result"
        assert expected in terminal.primary.message.casefold()
        if before is not None:
            assert target.read_bytes() == before
        assert lifecycle.phase is RunPhase.FAILED
        assert lifecycle.reset_permitted
        assert shell.run_controls.startButton.isEnabled()

        owner = custody["owner"]
        lease = custody["lease"]
        slot = custody.get("slot")
        assert owner.light_lease is None
        assert owner.light_slot is None
        assert lease.state is Light1DLeaseState.RELEASED
        if slot is not None:
            assert slot.state is Light1DCustodyState.CANCELLED
        assert lease.authority.snapshot().reservation_count == 0
        identity = terminal.run_identity
        retained_identity = page._context_controller.run_identity
        assert retained_identity is None or retained_identity is identity
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
        candidate.save_path = str(tmp_path / "valid.nexus")
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
