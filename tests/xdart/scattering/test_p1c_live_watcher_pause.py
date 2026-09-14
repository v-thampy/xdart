"""Finite P1-C Live-watcher Pause correction oracle."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from threading import Event, Thread
import time

import pytest

from tests.xdart.scattering._e2sd_support import write_poni
from tests.xdart.scattering.test_p1b_comprehensive_live import (
    _finish_live,
    _live_intent,
    _wait_physical_results,
    _write_tiff,
)
from tests.xdart.scattering.test_p1b_output_graph import (
    _bridge_legacy_expected_target_state,
    _drain_until,
    _start,
)
from xdart.gui.tabs.scattering.acquisition_runtime import AcquisitionRuntime
from xdart.gui.tabs.scattering.adapters.run_executor import (
    StandardRunExecutor,
    _StandardRun,
)
from xdart.gui.tabs.scattering.context_controller import ContextController
from xdart.gui.tabs.scattering.context_projection import ContextProjection
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.events import (
    CleanupStatus,
    DurablePaused,
    ExecutorAccepted,
    PreflightAccepted,
    RunIdentity,
)
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xrd_tools.session.run_configuration import RunIntent


def test_sessionless_live_pause_waits_for_effect_and_resume_reopens() -> None:
    identity = RunIdentity(1, "sessionless-live")
    runtime = AcquisitionRuntime()
    runtime._arm_live()
    effect_entered = Event()
    release_effect = Event()
    effect_finished = Event()

    def active_effect() -> None:
        with runtime._live_effect():
            effect_entered.set()
            assert release_effect.wait(2.0)
        effect_finished.set()

    worker = Thread(target=active_effect, name="live-effect")
    worker.start()
    assert effect_entered.wait(1.0)

    with pytest.raises(TimeoutError, match="Live effects"):
        runtime.pause(None, identity, 0.01, session_supplier=lambda: None)
    assert runtime._gate.is_set() and runtime._live_admitting

    paused: list[DurablePaused] = []
    pause_worker = Thread(
        target=lambda: paused.append(runtime.pause(
            None,
            identity,
            1.0,
            session_supplier=lambda: None,
        )),
        name="live-pause",
    )
    pause_worker.start()
    time.sleep(0.05)
    assert pause_worker.is_alive()
    assert paused == []

    release_effect.set()
    pause_worker.join(1.0)
    worker.join(1.0)
    assert not pause_worker.is_alive() and effect_finished.is_set()
    assert paused == [DurablePaused(identity, 1)]

    resumed_effect = Event()

    def post_pause_effect() -> None:
        with runtime._live_effect():
            resumed_effect.set()

    parked = Thread(target=post_pause_effect, name="parked-live-effect")
    parked.start()
    assert not resumed_effect.wait(0.05)
    runtime.resume(None)
    parked.join(1.0)
    assert resumed_effect.is_set()


def test_empty_live_context_pause_resume_does_not_select_missing_context() -> None:
    configuration = RunIntent().freeze()
    lifecycle = ScatteringCoordinator()
    request = lifecycle.begin_start().request_id
    accepted = lifecycle.preflight_accepted(
        PreflightAccepted(request, configuration)
    )
    identity = accepted.run_identity
    assert identity is not None
    assert lifecycle.executor_accepted(ExecutorAccepted(identity))

    class Executor:
        def pause(self, exact_identity):
            assert exact_identity is identity
            return DurablePaused(identity, 1)

        def resume(self, exact_identity) -> None:
            assert exact_identity is identity

    controller = ContextController(
        lifecycle=lifecycle,
        executor=Executor(),
        browse_loader=object(),
        projection=ContextProjection(),
    )

    assert controller.acquisition_context is None
    assert controller.pause() == DurablePaused(identity, 1)
    assert lifecycle.phase is RunPhase.PAUSED
    assert controller.selection is None
    assert controller.resume() is None
    assert lifecycle.phase is RunPhase.RUNNING
    assert controller.selection is None


def test_all_committed_live_append_pauses_without_touching_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xrd_tools.sources.directory_session import DirectoryIndexSession
    _bridge_legacy_expected_target_state(monkeypatch)
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    for label in range(1, 6):
        _write_tiff(source / f"prefix_{label:04d}.tif", label)
    seed = StandardRunExecutor(join_timeout=2.0)
    seed_identity = _start(
        seed, _live_intent(source, output, poni, suffixes=(".tif",)),
        request_value=1900,
    )
    seeded = _wait_physical_results(
        seed, seed_identity, (), expected_rows=tuple(range(1, 6)),
        expected_files=(5, 0, 0, 5), output_root=output, timeout=30.0,
    )
    _finish_live(seed, seed_identity, prior=seeded)
    paths = tuple(sorted(output.rglob("*.nexus"))) + tuple(
        sorted(output.rglob("*.xye"))
    )
    before = {path: (path.read_bytes(), path.stat().st_ino,
                     path.stat().st_mtime_ns) for path in paths}
    observations = 0
    observe = DirectoryIndexSession.observe
    def counted_observe(owner, *, refresh=True):
        nonlocal observations
        observations += 1
        return observe(owner, refresh=refresh)
    monkeypatch.setattr(DirectoryIndexSession, "observe", counted_observe)
    intent = _live_intent(source, output, poni, suffixes=(".tif",))
    intent.output_mode = "Append"
    executor = StandardRunExecutor(join_timeout=2.0)
    identity = _start(executor, intent, request_value=1902)
    _drain_until(executor, lambda values: any(
        event.kind is StandardEventKind.DISCOVERY
        and event.files_processed == 5 and event.files_pending == 0
        for event in values
    ))
    assert executor.pause(identity) == DurablePaused(identity, 1)
    run, baseline = executor._exact_run(identity), observations
    time.sleep(0.2)
    assert run is not None and run.session is None
    assert observations == baseline
    assert tuple(key.local_frame_label for key in
                 executor.frame_catalog(identity).entries) == tuple(range(1, 6))
    assert {path: (path.read_bytes(), path.stat().st_ino,
                   path.stat().st_mtime_ns) for path in paths} == before
    executor.resume(identity)
    deadline = time.monotonic() + 1.0
    while observations == baseline and time.monotonic() < deadline:
        time.sleep(0.01)
    assert observations > baseline
    _finish_live(executor, identity)


def test_pause_waits_for_live_activation_then_uses_installed_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xrd_tools.sources.directory_session import DirectoryIndexSession
    _bridge_legacy_expected_target_state(monkeypatch)
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    executor = StandardRunExecutor(join_timeout=2.0)
    observations = 0
    observe = DirectoryIndexSession.observe

    def counted_observe(owner, *, refresh=True):
        nonlocal observations
        observations += 1
        return observe(owner, refresh=refresh)
    monkeypatch.setattr(DirectoryIndexSession, "observe", counted_observe)
    entered, release = Event(), Event()
    construct = executor._construct

    def blocked_construct(run, *args, **kwargs):
        entered.set()
        assert release.wait(5.0)
        return construct(run, *args, **kwargs)

    monkeypatch.setattr(executor, "_construct", blocked_construct)
    identity = _start(
        executor,
        _live_intent(source, output, poni, suffixes=(".tif",)),
        request_value=1901,
    )
    prior = _drain_until(
        executor,
        lambda values: any(
            event.kind is StandardEventKind.DISCOVERY
            and event.files_discovered == 0
            for event in values
        ),
    )
    assert executor.pause(identity) == DurablePaused(identity, 1)
    baseline = observations
    _write_tiff(source / "late_0001.tif", 1)
    time.sleep(0.2)
    assert observations == baseline and not entered.is_set()
    assert tuple(output.rglob("*")) == ()
    executor.resume(identity)
    assert entered.wait(10.0)
    paused: list[DurablePaused] = []
    command = Thread(
        target=lambda: paused.append(executor.pause(identity)),
        name="activation-pause",
    )
    command.start()
    time.sleep(0.05)
    assert command.is_alive() and paused == []
    release.set()
    command.join(10.0)
    assert not command.is_alive()
    assert paused == [DurablePaused(identity, 2)]
    run = executor._exact_run(identity)
    assert run is not None and run.session is not None
    assert not any(
        event.kind is StandardEventKind.FRAME_READY
        for event in executor.drain_events()
    )
    executor.resume(identity)
    first = _wait_physical_results(
        executor,
        identity,
        prior,
        expected_rows=(1,),
        expected_files=(1, 0, 0, 1),
        output_root=output,
        timeout=30.0,
    )
    project = executor._project_new_durable
    settling, release_settlement = Event(), Event()

    def blocked_project(run, graph) -> None:
        settling.set()
        assert release_settlement.wait(5.0)
        project(run, graph)

    monkeypatch.setattr(executor, "_project_new_durable", blocked_project)
    _write_tiff(source / "late_0002.tif", 2)
    assert settling.wait(10.0)
    second_pause: list[DurablePaused] = []
    command = Thread(
        target=lambda: second_pause.append(executor.pause(identity)),
        name="settlement-pause",
    )
    command.start()
    time.sleep(0.05)
    assert command.is_alive() and second_pause == []
    release_settlement.set()
    command.join(10.0)
    assert second_pause == [DurablePaused(identity, 3)]
    executor.resume(identity)
    second = _wait_physical_results(
        executor,
        identity,
        first,
        expected_rows=(1, 2),
        expected_files=(2, 0, 0, 2),
        output_root=output,
        timeout=30.0,
    )
    assert executor.pause(identity) == DurablePaused(identity, 4)
    assert executor.close(identity).cleanup_status is CleanupStatus.CLEANED
    terminal = _drain_until(executor, lambda values: any(
        event.kind is StandardEventKind.STOPPED for event in values
    ))
    assert next(event for event in terminal if
                event.kind is StandardEventKind.STOPPED).cleanup_status \
        is CleanupStatus.CLEANED
    assert executor.close(identity).cleanup_status is CleanupStatus.CLEANED


def test_live_resume_failure_keeps_exact_session_and_effects_paused() -> None:
    class Session:
        def __init__(self) -> None:
            self.pause_calls = self.resume_calls = 0

        def pause(self, *, timeout: float) -> bool:
            self.pause_calls += 1
            return timeout >= 0.0

        def flush(self, *, force: bool) -> None:
            assert force and self.pause_calls

        def resume(self) -> None:
            self.resume_calls += 1
            if self.resume_calls == 1:
                raise RuntimeError("retry resume")

    identity = RunIdentity(2, "retry-live")
    runtime, session = AcquisitionRuntime(), Session()
    runtime._arm_live()
    projected: list[str] = []
    assert runtime.pause(
        session,
        identity,
        1.0,
        drain_projection=lambda _timeout: not projected.append("drained"),
    ) == DurablePaused(identity, 1)
    parked, effect_error = Event(), []

    def next_effect() -> None:
        try:
            with runtime._live_effect():
                parked.set()
        except BaseException as error:
            effect_error.append(error)

    worker = Thread(target=next_effect, name="post-pause-effect")
    worker.start()
    assert not parked.wait(0.05)
    with pytest.raises(RuntimeError, match="retry resume"):
        runtime.resume(session)
    assert session.pause_calls == 2
    assert not parked.wait(0.05)
    runtime.resume(session)
    worker.join(1.0)
    assert parked.is_set() and effect_error == []
    assert projected == ["drained"]
    runtime.request_stop(None)


def test_stop_wins_pending_pause_and_close_stops_before_retirement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = AcquisitionRuntime()
    runtime._arm_live()
    entered, release = Event(), Event()

    def blocked_effect() -> None:
        with runtime._live_effect():
            entered.set()
            assert release.wait(2.0)

    effect = Thread(target=blocked_effect, name="stop-race-effect")
    effect.start()
    assert entered.wait(1.0)
    failures: list[BaseException] = []
    pause = Thread(
        target=lambda: _capture_failure(
            failures, runtime.pause, None, RunIdentity(3, "stop-race"), 5.0
        ),
        name="stop-race-pause",
    )
    pause.start()
    time.sleep(0.05)
    runtime.request_stop(None)
    pause.join(0.5)
    assert not pause.is_alive()
    assert len(failures) == 1 and "stopped" in str(failures[0])
    release.set()
    effect.join(1.0)
    with pytest.raises(RuntimeError, match="admission cancelled"):
        with runtime._live_effect():
            pass
    runtime.request_stop(None)

    order: list[str] = []
    identity = RunIdentity(4, "close-order")
    run = _StandardRun(None, identity, None, None, None, None, tmp_path / "x")
    run.closed = True
    close_runtime = AcquisitionRuntime()
    close_runtime._arm_live()
    late_context = SimpleNamespace(retire=lambda: order.append("late-context"))
    activated, release_activation = Event(), Event()

    def late_activation() -> None:
        with close_runtime._live_effect():
            activated.set()
            assert release_activation.wait(2.0)
            close_runtime.context = late_context

    run.worker = Thread(target=late_activation, name="close-race-activation")
    run.worker.start()
    assert activated.wait(1.0)
    retire = close_runtime.retire

    def retire_then_release() -> None:
        order.append("retire")
        retire()
        release_activation.set()

    close_runtime.retire = retire_then_release
    run.context_runtime = close_runtime
    run.display = SimpleNamespace(retire=lambda **_kwargs: order.append("display") or True)
    executor = StandardRunExecutor()
    executor._active = run
    monkeypatch.setattr(executor, "stop", lambda exact: order.append("stop"))
    assert executor._close_run(identity).cleanup_status is CleanupStatus.CLEANED
    assert order == ["stop", "retire", "display", "late-context"]
    assert not run.worker.is_alive()


def _capture_failure(target: list[BaseException], operation, *args) -> None:
    try:
        operation(*args)
    except BaseException as error:
        target.append(error)
