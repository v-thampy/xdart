from __future__ import annotations

from pathlib import Path
from threading import Event, Thread
import time

from pyqtgraph.Qt import QtWidgets

from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent, ThresholdIntent
from xrd_tools.sources.selection import image_series_spec
from xdart.gui.tabs.scattering.adapters import run_executor as executor_module
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor, _StandardRun
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.contracts import SourceCapture
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xdart.gui.tabs.scattering.events import (
    CleanupStatus,
    ExecutorAccepted,
    PreflightAccepted,
    RequestId,
    RunIdentity,
)
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.state_machine import RunPhase
from tests.xdart.scattering._admission import install_admission


class _PendingSession:
    def __init__(self) -> None:
        self.stop_calls = 0

    def stop(self) -> None:
        self.stop_calls += 1


class _OpenSource:
    def __init__(self, *, raise_on_close: bool = False) -> None:
        self.close_calls = 0
        self.raise_on_close = raise_on_close

    def close(self) -> None:
        self.close_calls += 1
        if self.raise_on_close:
            raise RuntimeError("source close failed")


def _active_lifecycle(configuration):
    lifecycle = ScatteringCoordinator()
    begun = lifecycle.begin_start()
    promoted = lifecycle.preflight_accepted(
        PreflightAccepted(begun.request_id, configuration)
    )
    identity = promoted.run_identity
    assert identity is not None
    accepted = lifecycle.executor_accepted(ExecutorAccepted(identity))
    assert accepted.phase is RunPhase.RUNNING
    return lifecycle, identity


def _terminal(executor: StandardRunExecutor):
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


def test_page_close_does_not_ack_owners_while_executor_worker_is_alive() -> None:
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    configuration = RunIntent().freeze()
    lifecycle, identity = _active_lifecycle(configuration)
    executor = StandardRunExecutor(join_timeout=0.01)
    session = _PendingSession()
    source = _OpenSource()
    release = Event()
    worker = Thread(target=release.wait, daemon=True)
    run = _StandardRun(
        configuration,
        identity,
        object(),
        source,
        session,
        None,
        Path("pending.nxs"),
        worker=worker,
    )
    executor._active = run
    worker.start()
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent()),
        lifecycle=lifecycle,
        sources=FilesystemSourceAdapter(),
        executor=executor,
    )
    try:
        page.close_workspace()
        qapp.processEvents()
        assert lifecycle.phase is RunPhase.STOPPING
        assert lifecycle.closed is True
        assert run.closed is False
        assert run.session is session
        assert source.close_calls == 0
    finally:
        release.set()
        worker.join(timeout=1)
        page.deleteLater()
        qapp.processEvents()


def test_start_cleanup_failure_is_not_reported_as_cleaned(monkeypatch, tmp_path) -> None:
    selected = image_series_spec(tmp_path / "raw_0001.tif")
    configuration = RunIntent(
        source_spec=selected,
        poni_file=str(tmp_path / "bad.poni"),
        save_path=str(tmp_path / "out.nxs"),
        output_mode="Overwrite",
    ).freeze()
    identity = RunIdentity.from_configuration(configuration)
    opened = _OpenSource(raise_on_close=True)
    monkeypatch.setattr(executor_module, "open_source", lambda _spec: opened)
    monkeypatch.setattr(
        executor_module,
        "load_poni",
        lambda _path: (_ for _ in ()).throw(RuntimeError("invalid PONI")),
    )

    executor = StandardRunExecutor()
    capture = SourceCapture(RequestId(1), 1, selected)
    admission = install_admission(executor, configuration, capture)
    result = executor.start(
        configuration,
        capture,
        identity,
        admission,
    )

    assert isinstance(result, ExecutorAccepted)
    terminal = _terminal(executor)
    assert terminal.kind is StandardEventKind.FAILED
    assert terminal.cleanup_status is CleanupStatus.CLEANUP_PENDING
    assert opened.close_calls == 1


def test_disabled_threshold_does_not_reach_reduction_plan(monkeypatch, tmp_path) -> None:
    from tests.xdart.scattering._e2sd_support import write_poni
    from tests.xdart.scattering.test_p1b_output_graph import _intent, _write_tiff
    from tests.xdart.scattering.test_e1b1_terminal_cleanup import _prepared_run

    raw, poni = tmp_path / "raw_0001.tif", tmp_path / "cal.poni"
    _write_tiff(raw, 1)
    write_poni(poni)
    intent = _intent(raw, tmp_path / "processed", poni)
    intent.threshold = ThresholdIntent(
        apply_threshold=False, threshold_min=10.0, threshold_max=20.0,
    )
    executor, run, _admission = _prepared_run(tmp_path, intent=intent)
    plans = []
    real_plan = executor_module.native_int_reduction_plan

    def capture_plan(configuration, **options):
        plan = real_plan(configuration, **options)
        plans.append(plan)
        return plan

    monkeypatch.setattr(executor_module, "native_int_reduction_plan", capture_plan)
    executor._run(run)
    terminal = _terminal(executor)
    assert terminal.kind is StandardEventKind.FINISHED, terminal.detail
    assert len(plans) == 1
    assert plans[0].threshold_min is None
    assert plans[0].threshold_max is None
    assert executor.close(run.identity).cleanup_status is CleanupStatus.CLEANED
