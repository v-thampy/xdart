from __future__ import annotations

import logging
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from xrd_tools.core.scan import SourceKind
from xrd_tools.session.run_configuration import RunIntent
from xdart.gui.tabs.scattering.adapters import run_executor as executor_module
from xdart.gui.tabs.scattering.adapters.run_executor import (
    StandardRunExecutor,
    _StandardRun,
    _candidate_file_owners,
    _claim_output_physical_files,
    _eager_directory_file_counts,
)
from xdart.gui.tabs.scattering.acquisition_runtime import AcquisitionRuntime
from xdart.gui.tabs.scattering.display_values import (
    StandardEventKind,
    StandardRunEvent,
)
from xdart.gui.tabs.scattering.events import (
    CleanupStatus,
    ExecutorClosed,
    PauseFailed,
    RunIdentity,
)
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.state_machine import RunPhase

from tests.xdart.scattering.test_e3_context_contract import (
    _running_controller,
)


def test_linked_candidate_keeps_its_own_directory_counter_slot(
    tmp_path: Path,
) -> None:
    master = tmp_path / "master.h5"
    linked_candidate = tmp_path / "linked.h5"
    sidecar = tmp_path / "master_data.h5"
    discovered = (master, linked_candidate, sidecar)

    def container_item(
        source_path: Path,
        member_paths: tuple[Path, ...],
    ) -> SimpleNamespace:
        return SimpleNamespace(
            source_path=source_path,
            source_spec=SimpleNamespace(kind=SourceKind.NEXUS_STACK),
            source_stamp=SimpleNamespace(
                members=(),
                external_members=(),
            ),
            group=SimpleNamespace(member_paths=member_paths),
        )

    first = container_item(master, discovered)
    second = container_item(linked_candidate, (linked_candidate,))
    outputs = (
        SimpleNamespace(item=first),
        SimpleNamespace(item=second),
    )

    # The standalone sidecar belongs to its consumer, but a dependency that
    # is also another selected candidate remains reserved for that candidate.
    assert _eager_directory_file_counts(outputs, discovered) == ((2, 1), 0)

    owners = _candidate_file_owners(((master,), (linked_candidate,)))
    claimed: set[Path] = set()
    assert _claim_output_physical_files(
        first,
        discovered,
        claimed,
        candidate_owners=owners,
        output_index=0,
    ) == 2
    assert linked_candidate not in claimed
    assert _claim_output_physical_files(
        second,
        discovered,
        claimed,
        candidate_owners=owners,
        output_index=1,
    ) == 1
    assert claimed == set(discovered)


def test_stopped_deferred_skip_counts_known_unowned_sidecar(
    monkeypatch,
    tmp_path: Path,
) -> None:
    master = tmp_path / "master.h5"
    linked_candidate = tmp_path / "linked.h5"
    sidecar = tmp_path / "master_data.h5"
    first = SimpleNamespace(
        physical_paths=(master,),
        protected_states=(
            SimpleNamespace(path=str(master)),
            SimpleNamespace(path=str(sidecar)),
        ),
    )
    second = SimpleNamespace(
        physical_paths=(linked_candidate,),
        protected_states=(SimpleNamespace(path=str(linked_candidate)),),
    )
    deferred = SimpleNamespace(
        entries=(first, second),
        discovered_paths=(master, linked_candidate, sidecar),
        discovered_file_count=3,
    )
    configuration = RunIntent().freeze()
    identity = RunIdentity.from_configuration(configuration)
    run = _StandardRun(
        configuration,
        identity,
        None,
        None,
        None,
        None,
        Path("initial.nexus"),
        resources=SimpleNamespace(directory_session=object()),
    )
    executor = StandardRunExecutor()

    def skip_then_stop(*_args, **_kwargs):
        run.stop_requested = True
        return None, 0, 1

    monkeypatch.setattr(
        executor_module,
        "materialize_deferred_output",
        skip_then_stop,
    )

    assert executor._execute_deferred_directory(
        run,
        SimpleNamespace(),
        deferred,
    ) is True
    event = executor.drain_events()[-1]
    assert (
        event.files_processed,
        event.files_skipped,
        event.files_pending,
        event.files_discovered,
    ) == (0, 2, 1, 3)


@pytest.mark.parametrize(
    ("stopped", "failure", "expected"),
    (
        (False, False, StandardEventKind.FINISHED),
        (True, False, StandardEventKind.STOPPED),
        (False, True, StandardEventKind.FAILED),
    ),
)
def test_terminal_summary_is_measured_and_emitted_exactly_once(
    monkeypatch,
    caplog,
    stopped: bool,
    failure: bool,
    expected: StandardEventKind,
) -> None:
    configuration = RunIntent(max_cores=3).freeze()
    identity = RunIdentity.from_configuration(configuration)
    run = _StandardRun(
        configuration,
        identity,
        None,
        None,
        None,
        None,
        Path("measured-output.nxs"),
        completed=4,
        total=5,
    )
    executor = StandardRunExecutor()
    run.perf_enabled = True
    run.perf_values = {
        "source_read": 0.5,
        "submit_wait": 1.5,
        "sink_nexus_write": 0.25,
        "sink_nexus_flush": 0.125,
        "sink_xye_write": 0.75,
        "finish_wait": 0.375,
        "display_callback": 0.05,
        "display_projection": 0.075,
    }

    def execute(_run: _StandardRun) -> bool:
        if failure:
            raise RuntimeError("measured failure")
        return stopped

    def cleanup(
        _run: _StandardRun,
        primary=None,
    ) -> ExecutorClosed:
        return ExecutorClosed(
            identity,
            CleanupStatus.CLEANED,
            primary=primary,
        )

    clock = iter((10.0, 12.5, 12.6, 13.0))
    monkeypatch.setattr(executor_module, "monotonic", lambda: next(clock))
    monkeypatch.setattr(executor, "_execute_admitted", execute)
    monkeypatch.setattr(executor, "_cleanup", cleanup)
    caplog.set_level(logging.INFO, logger=executor_module.__name__)

    executor._run(run)
    terminal = executor.drain_events()

    assert len(terminal) == 1
    assert terminal[0].kind is expected
    assert terminal[0].completed == 4
    assert terminal[0].total == 5
    assert terminal[0].terminal_timing is not None
    assert terminal[0].terminal_timing.elapsed_seconds == 3.0
    assert terminal[0].terminal_timing.work_seconds == 2.5
    assert terminal[0].terminal_timing.cleanup_seconds == pytest.approx(0.4)
    assert dict(terminal[0].terminal_timing.details) == {
        "source_read": 0.5,
        "submit_wait": 1.5,
        "writer_batch": 0.25,
        "writer_flush": 0.125,
        "xye": 0.75,
        "finish_wait": 0.375,
        "display": pytest.approx(0.125),
    }

    executor._terminal_event(
        run,
        expected,
        cleanup(run),
        4,
        5,
        elapsed=99.0,
        work_elapsed=98.0,
        cleanup_elapsed=1.0,
        core_count=9,
    )
    assert executor.drain_events() == ()

    messages = [record.getMessage() for record in caplog.records]
    assert messages.count("Total Frames Processed: 4") == 1
    assert messages.count("Total Time: 3.00s") == 1
    summaries = [
        message for message in messages if message.startswith("[PERF-SUMMARY]")
    ]
    assert summaries == [
        (
            f"[PERF-SUMMARY] outcome={expected.value} frames=4/5 cores=3 | "
            "total=3.00s work=2.50s cleanup=0.40s | "
            "throughput=1.3 frames/s | output=measured-output.nxs"
        )
    ]


@pytest.mark.parametrize(
    ("total", "expected_counts"),
    (
        (651, (163, 163, 163, 162)),
        (3_621, (906, 905, 905, 905)),
    ),
)
def test_quartile_capture_uses_exact_cumulative_boundaries_and_terminal_tail(
    total: int,
    expected_counts: tuple[int, int, int, int],
) -> None:
    capture_type = getattr(executor_module, "_RunQuartileCapture")
    capture = capture_type(total=total, started_at=0.0)
    boundaries = []
    for completed in range(1, total + 1):
        boundary = capture.observe(
            completed,
            now=completed / 10.0,
            cumulative={
                "reducer_compute": float(completed) / 100.0,
                "reducer_compute_count": float(completed),
                "source_read": float(completed),
                "submit_wait": float(completed * 2),
                "sink_nexus_write": float(completed * 3),
                "sink_nexus_flush": float(completed * 4),
                "sink_xye_write": float(completed * 5),
                "sink_xye_promotion": float(completed),
                "session_record_upsert": float(completed),
                "session_frame_listeners": float(completed * 2),
                "session_progress_listeners": float(completed * 3),
                "display_projection": float(completed * 4),
                "finish_wait": 0.0,
            },
        )
        if boundary is not None:
            boundaries.append(boundary)
    timing = capture.finish(
        completed=total,
        now=total / 10.0,
        cumulative={
            "reducer_compute": float(total) / 100.0,
            "reducer_compute_count": float(total),
            "source_read": float(total),
            "submit_wait": float(total * 2),
            "sink_nexus_write": float(total * 3),
            "sink_nexus_flush": float(total * 4),
            "sink_xye_write": float(total * 5),
            "sink_xye_promotion": float(total),
            "session_record_upsert": float(total),
            "session_frame_listeners": float(total * 2),
            "session_progress_listeners": float(total * 3),
            "display_projection": float(total * 4),
            "finish_wait": 7.0,
        },
    )

    assert capture.frame_counts == expected_counts
    assert [boundary.quartile for boundary in boundaries] == [1, 2, 3]
    assert timing is not None
    assert timing.frame_counts == expected_counts
    assert timing.compute_counts == expected_counts
    details = dict(timing.details)
    assert details["reducer_compute"] == pytest.approx(
        tuple(float(count) / 100.0 for count in expected_counts)
    )
    assert details["source_read"] == pytest.approx(
        tuple(float(count) for count in expected_counts)
    )
    assert details["xye"] == pytest.approx(
        tuple(float(count * 6) for count in expected_counts)
    )
    assert details["completion_display"] == pytest.approx(
        tuple(float(count * 10) for count in expected_counts)
    )
    assert details["finish_wait"] == pytest.approx((0.0, 0.0, 0.0, 7.0))


def test_unmeasured_projection_failure_omits_false_zero_timing(
    caplog,
) -> None:
    configuration = RunIntent().freeze()
    identity = RunIdentity.from_configuration(configuration)
    run = _StandardRun(
        configuration,
        identity,
        None,
        None,
        None,
        None,
        Path("projection-failed.nxs"),
    )
    diagnostic = executor_module.detach_exception(
        RuntimeError("projection failed"), "display_projection",
    )
    executor = StandardRunExecutor()
    caplog.set_level(logging.INFO, logger=executor_module.__name__)
    executor._terminal_event(
        run,
        StandardEventKind.FAILED,
        ExecutorClosed(
            identity,
            CleanupStatus.CLEANED,
            primary=diagnostic,
        ),
        0,
        1,
    )
    event = executor.drain_events()[0]
    assert event.terminal_timing is None
    assert event.detail == "projection failed"
    messages = [record.getMessage() for record in caplog.records]
    assert "Total Time: unavailable" in messages
    assert any(
        message.startswith("[PERF-SUMMARY]")
        and "total=unavailable" in message
        for message in messages
    )


def test_failed_terminal_folds_a_durably_completed_container_file(
    monkeypatch,
) -> None:
    configuration = RunIntent().freeze()
    identity = RunIdentity.from_configuration(configuration)
    run = _StandardRun(
        configuration,
        identity,
        None,
        None,
        None,
        None,
        Path("container.nexus"),
        files_discovered=1,
        current_file_total=1,
        current_total=2,
    )
    executor = StandardRunExecutor()

    def execute(current: _StandardRun) -> bool:
        current.completed = 2
        current.total = 2
        current.current_completed = 2
        current.current_published = 2
        raise RuntimeError("display projection failed")

    def cleanup(
        _run: _StandardRun,
        primary=None,
    ) -> ExecutorClosed:
        return ExecutorClosed(
            identity,
            CleanupStatus.CLEANED,
            primary=primary,
        )

    monkeypatch.setattr(executor, "_execute_admitted", execute)
    monkeypatch.setattr(executor, "_cleanup", cleanup)

    executor._run(run)
    terminal = executor.drain_events()

    assert len(terminal) == 1
    event = terminal[0]
    assert event.kind is StandardEventKind.FAILED
    assert (event.artifact_completed, event.artifact_total) == (2, 2)
    assert (
        event.files_processed,
        event.files_skipped,
        event.files_pending,
        event.files_discovered,
    ) == (1, 0, 0, 1)


def test_projection_failure_keeps_durable_artifact_and_frame_progress(
    monkeypatch,
) -> None:
    configuration = RunIntent().freeze()
    identity = RunIdentity.from_configuration(configuration)

    class Session:
        def start(self) -> None:
            return None

    class Source:
        def close(self) -> None:
            raise AssertionError("failure cleanup, not success, owns close")

    run = _StandardRun(
        configuration,
        identity,
        SimpleNamespace(frames=()),
        Source(),
        Session(),
        None,
        Path("durable.nexus"),
        current_total=2,
        files_discovered=1,
        current_file_total=1,
    )
    executor = StandardRunExecutor()
    monkeypatch.setattr(
        executor,
        "_finish_session",
        lambda _run: SimpleNamespace(
            n_processed=2,
            failed=False,
            cancelled=False,
        ),
    )
    monkeypatch.setattr(
        executor,
        "_finish_display_projection",
        lambda _run: (_ for _ in ()).throw(
            RuntimeError("projection failed")
        ),
    )

    with pytest.raises(RuntimeError, match="projection failed"):
        executor._execute_current(run, construct=False)

    assert (run.completed, run.current_completed) == (2, 2)
    assert run.current_published == 0
    assert run.artifacts == [Path("durable.nexus")]

    executor._terminal_event(
        run,
        StandardEventKind.FAILED,
        ExecutorClosed(identity, CleanupStatus.CLEANED),
        2,
        2,
    )
    event = executor.drain_events()[0]
    assert event.artifacts == ("durable.nexus",)
    assert (event.artifact_completed, event.artifact_total) == (2, 2)
    assert (
        event.files_processed,
        event.files_skipped,
        event.files_pending,
        event.files_discovered,
    ) == (1, 0, 0, 1)


def test_terminal_durable_progress_does_not_invent_published_navigation() -> None:
    configuration = RunIntent().freeze()
    identity = RunIdentity.from_configuration(configuration)
    owner = SimpleNamespace(_artifact_progress={})
    artifact = "durable.nexus"

    ScatteringWorkspace._record_artifact_progress(
        owner,
        StandardRunEvent(
            identity,
            StandardEventKind.FRAME_READY,
            artifact=artifact,
            artifact_completed=10,
            artifact_total=1000,
        ),
    )
    ScatteringWorkspace._record_artifact_progress(
        owner,
        StandardRunEvent(
            identity,
            StandardEventKind.FAILED,
            artifact=artifact,
            artifact_completed=1000,
            artifact_total=1000,
        ),
    )

    progress = owner._artifact_progress[artifact]
    assert (progress.completed, progress.published, progress.total) == (
        1000,
        10,
        1000,
    )


def test_failed_session_does_not_report_unfinalized_artifact(
    monkeypatch,
) -> None:
    configuration = RunIntent().freeze()
    identity = RunIdentity.from_configuration(configuration)

    class Session:
        def start(self) -> None:
            return None

    run = _StandardRun(
        configuration,
        identity,
        SimpleNamespace(frames=()),
        SimpleNamespace(),
        Session(),
        None,
        Path("aborted.nexus"),
        current_total=2,
    )
    executor = StandardRunExecutor()
    monkeypatch.setattr(
        executor,
        "_finish_session",
        lambda _run: SimpleNamespace(
            n_processed=1,
            failed=True,
            cancelled=False,
            error="writer failed",
        ),
    )
    monkeypatch.setattr(
        executor,
        "_finish_display_projection",
        lambda _run: None,
    )

    with pytest.raises(RuntimeError, match="writer failed"):
        executor._execute_current(run, construct=False)

    assert run.artifacts == []


def test_second_eager_construct_failure_keeps_new_artifact_pending(
    monkeypatch,
) -> None:
    configuration = RunIntent(poni_file="accepted.poni", save_path="initial.nexus").freeze()
    identity = RunIdentity.from_configuration(configuration)

    def item(name: str, count: int):
        return SimpleNamespace(
            target=Path(name),
            source_spec=SimpleNamespace(kind=SourceKind.NEXUS_STACK),
            descriptor=None,
            source_stamp=SimpleNamespace(
                frame_count=count,
                members=(),
                external_members=(),
            ),
        )

    first = item("first.nexus", 5)
    second = item("second.nexus", 2)
    receipt = SimpleNamespace(
        outputs=(
            SimpleNamespace(item=first, labels=tuple(range(5))),
            SimpleNamespace(item=second, labels=tuple(range(2))),
        ),
        deferred_directory=None,
        directory_discovered_file_count=2,
    )
    run = _StandardRun(
        configuration,
        identity,
        None,
        None,
        None,
        None,
        Path("initial.nexus"),
        capture=SimpleNamespace(),
        resources=SimpleNamespace(
            admission=receipt,
            directory_session=None,
        ),
    )
    executor = StandardRunExecutor()
    real_construct = executor._construct
    drift = executor_module.SourceRevisionChanged("second source changed")
    validations = []

    def construct(current, *, item, labels, decision):
        if item is first:
            current.artifact = item.target
            current.current_completed = 0
            current.current_published = 0
            return current
        assert item is second
        current.current_completed = 0
        current.current_published = 0
        return real_construct(
            current,
            item=item,
            labels=labels,
            decision=decision,
        )

    def reject_second(attempted, *, cancelled):
        validations.append(attempted)
        assert cancelled() is False
        raise drift

    def execute_current(current, *, construct=False) -> bool:
        assert construct is False
        current.completed += current.current_total
        current.current_completed = current.current_total
        current.current_published = current.current_total
        if current.artifact not in current.artifacts:
            current.artifacts.append(current.artifact)
        return False

    monkeypatch.setattr(
        executor_module,
        "AdmissionReceipt",
        SimpleNamespace,
    )
    monkeypatch.setattr(
        executor_module,
        "validate_admitted_receipt",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        executor_module,
        "validate_planned_source",
        reject_second,
    )
    monkeypatch.setattr(executor, "_construct", construct)
    monkeypatch.setattr(executor, "_execute_current", execute_current)
    monkeypatch.setattr(
        executor,
        "_cleanup",
        lambda _run, primary=None: ExecutorClosed(
            identity,
            CleanupStatus.CLEANED,
            primary=primary,
        ),
    )

    executor._run(run)
    event = executor.drain_events()[-1]

    assert event.kind is StandardEventKind.FAILED
    assert event.artifact == "second.nexus"
    assert len(validations) == 1 and validations[0] is second
    assert event.primary is not None
    assert event.primary.type_qualname == type(drift).__qualname__
    assert event.primary.message == str(drift)
    assert event.artifacts == ("first.nexus",)
    assert (event.completed, event.total) == (5, 7)
    assert (event.artifact_completed, event.artifact_total) == (0, 2)
    assert (
        event.files_processed,
        event.files_skipped,
        event.files_pending,
        event.files_discovered,
    ) == (1, 0, 1, 2)


@pytest.mark.parametrize(
    ("incremental", "file_total", "expected"),
    (
        (True, 4, (2, 0, 2, 4)),
        (False, 1, (0, 0, 1, 1)),
    ),
)
def test_stopped_terminal_counts_members_but_not_partial_container(
    incremental: bool,
    file_total: int,
    expected: tuple[int, int, int, int],
) -> None:
    configuration = RunIntent().freeze()
    identity = RunIdentity.from_configuration(configuration)
    run = _StandardRun(
        configuration,
        identity,
        None,
        None,
        None,
        None,
        Path("partial.nexus"),
        current_total=4,
        current_completed=2,
        current_published=2,
        files_discovered=file_total,
        current_file_total=file_total,
        current_files_incremental=incremental,
    )
    executor = StandardRunExecutor()

    executor._terminal_event(
        run,
        StandardEventKind.STOPPED,
        ExecutorClosed(identity, CleanupStatus.CLEANED),
        2,
        4,
    )
    event = executor.drain_events()[0]

    assert (event.artifact_completed, event.artifact_total) == (2, 4)
    assert (
        event.files_processed,
        event.files_skipped,
        event.files_pending,
        event.files_discovered,
    ) == expected


def test_display_projection_leaves_hdf5_completion_thread(
    monkeypatch,
) -> None:
    configuration = RunIntent().freeze()
    identity = RunIdentity.from_configuration(configuration)
    session = object()
    run = _StandardRun(
        configuration,
        identity,
        None,
        None,
        session,
        None,
        Path("display-worker.nxs"),
    )
    executor = StandardRunExecutor()
    entered = threading.Event()
    release = threading.Event()
    worker_names: list[str] = []

    def project(_run, _event, _image, _session) -> None:
        assert _session is session
        worker_names.append(threading.current_thread().name)
        entered.set()
        assert release.wait(timeout=2.0)

    monkeypatch.setattr(executor, "_frame_ready_owned", project)
    executor._start_display_projection(run)

    executor._frame_ready(run, SimpleNamespace(frame_index=1))

    assert entered.wait(timeout=2.0)
    assert worker_names == ["scattering-display-projection"]
    release.set()
    executor._finish_display_projection(run)


class _PauseSession:
    def __init__(self) -> None:
        self.pause_calls: list[float] = []
        self.resumes = 0
        self.submits = 0
        self.stops = 0

    def pause(self, *, timeout: float) -> bool:
        self.pause_calls.append(timeout)
        return True

    def resume(self) -> None:
        self.resumes += 1

    def submit(self, _frame) -> bool:
        self.submits += 1
        return True

    def stop(self) -> None:
        self.stops += 1

    def finish(self, *, raise_on_failure: bool = False):
        assert raise_on_failure is False
        return SimpleNamespace(failed=False, n_processed=0)


def _projection_pause_run(
    executor: StandardRunExecutor,
    *,
    identity: RunIdentity | None = None,
) -> tuple[_StandardRun, _PauseSession]:
    configuration = RunIntent().freeze()
    identity = identity or RunIdentity.from_configuration(configuration)
    session = _PauseSession()
    run = _StandardRun(
        configuration,
        identity,
        None,
        None,
        session,
        None,
        Path("pause-projection.nexus"),
        context_runtime=AcquisitionRuntime(),
    )
    executor._active = run
    executor._start_display_projection(run)
    return run, session


def test_durable_pause_waits_for_accepted_display_projection_without_retiring_worker(
    monkeypatch,
) -> None:
    executor = StandardRunExecutor(join_timeout=0.5)
    run, session = _projection_pause_run(executor)
    entered = threading.Event()
    release = threading.Event()
    returned = threading.Event()
    published: list[object] = []
    result: list[object] = []
    worker = run.display_projection_worker

    def blocked_projection(_run, event, _image, _session) -> None:
        entered.set()
        assert release.wait(timeout=2.0)
        published.append(event)

    monkeypatch.setattr(executor, "_frame_ready_owned", blocked_projection)
    event = SimpleNamespace(frame_index=1)
    executor._frame_ready(run, event)
    assert entered.wait(timeout=1.0)

    def pause() -> None:
        result.append(executor.pause(run.identity))
        returned.set()

    command = threading.Thread(target=pause, name="pause-command")
    command.start()
    assert not returned.wait(timeout=0.05)
    assert published == []
    assert run.display_projection_worker is worker
    assert worker is not None and worker.is_alive()

    release.set()
    command.join(timeout=1.0)
    assert not command.is_alive()
    assert returned.is_set()
    assert result[0].run_identity is run.identity
    assert published == [event]
    frozen = tuple(published)
    assert returned.wait(timeout=0.02)
    assert tuple(published) == frozen
    assert run.display_projection_worker is worker
    assert worker is not None and worker.is_alive()

    run.context_runtime.resume(session)
    executor._finish_display_projection(run)


def test_projection_pause_timeout_compensates_with_live_worker_and_no_duplicate_keys(
    monkeypatch,
) -> None:
    executor = StandardRunExecutor(join_timeout=0.02)
    run, session = _projection_pause_run(executor)
    entered = threading.Event()
    release = threading.Event()
    projected: list[int] = []

    def project(_run, event, _image, _session) -> None:
        entered.set()
        if int(event.frame_index) == 1:
            assert release.wait(timeout=2.0)
        projected.append(int(event.frame_index))

    monkeypatch.setattr(executor, "_frame_ready_owned", project)
    executor._frame_ready(run, SimpleNamespace(frame_index=1))
    assert entered.wait(timeout=1.0)

    with pytest.raises(
        TimeoutError,
        match="display projection did not reach durable pause",
    ):
        executor.pause(run.identity)
    assert session.resumes == 1
    assert run.context_runtime.submit(session, object()) is True
    assert session.submits == 1
    worker = run.display_projection_worker
    assert worker is not None and worker.is_alive()

    release.set()
    executor._frame_ready(run, SimpleNamespace(frame_index=2))
    assert executor._drain_display_projection(run, 1.0) is True
    assert projected == [1, 2]
    assert len(projected) == len(set(projected))
    assert run.display_projection_worker is worker
    assert worker.is_alive()
    executor._finish_display_projection(run)


def test_projection_exception_is_terminal_failed_and_cleanup_is_visible(
    monkeypatch,
) -> None:
    controller, lifecycle, mounted, _, _ = _running_controller()
    executor = StandardRunExecutor(join_timeout=0.1)
    run, session = _projection_pause_run(
        executor,
        identity=mounted.identity,
    )
    controller._executor = executor
    entered = threading.Event()
    primary = RuntimeError("queued display projection failed")

    def fail_projection(_run, _event, _image, _session) -> None:
        entered.set()
        raise primary

    monkeypatch.setattr(executor, "_frame_ready_owned", fail_projection)
    executor._frame_ready(run, SimpleNamespace(frame_index=1))
    assert entered.wait(timeout=1.0)

    failed = controller.pause()
    assert type(failed) is PauseFailed
    assert failed.diagnostic.message == str(primary)
    assert failed.diagnostic.operation == "context.pause"
    assert lifecycle.phase is RunPhase.FAILED
    assert session.resumes == 0
    assert session.stops == 1
    assert run.context_runtime is not None
    assert run.context_runtime._gate.is_set() is False
    assert run.closed is True
    assert run.display_projection_worker is None
    terminal = executor.drain_events()
    assert len(terminal) == 1
    assert terminal[0].kind is StandardEventKind.FAILED
    assert terminal[0].primary is not None
    assert terminal[0].primary.message == str(primary)
    assert terminal[0].cleanup_status is CleanupStatus.CLEANED


def test_dead_projection_worker_without_error_is_terminal_failed(
    monkeypatch,
) -> None:
    controller, lifecycle, mounted, _, _ = _running_controller()
    executor = StandardRunExecutor(join_timeout=0.1)
    run, session = _projection_pause_run(
        executor,
        identity=mounted.identity,
    )
    controller._executor = executor
    pending = run.display_projection_queue
    worker = run.display_projection_worker
    assert pending is not None and worker is not None
    pending.put(executor_module._DISPLAY_PROJECTION_END)
    worker.join(timeout=1.0)
    assert worker.is_alive() is False
    assert run.display_projection_errors == []

    failed = controller.pause()
    assert type(failed) is PauseFailed
    assert failed.diagnostic.message == (
        "display projection worker stopped before durable pause"
    )
    assert lifecycle.phase is RunPhase.FAILED
    assert session.resumes == 0
    assert session.stops == 1
    assert run.context_runtime is not None
    assert run.context_runtime._gate.is_set() is False
    assert run.closed is True
    assert run.display_projection_worker is None
    terminal = executor.drain_events()
    assert len(terminal) == 1
    assert terminal[0].kind is StandardEventKind.FAILED
    assert terminal[0].primary is not None
    assert terminal[0].primary.message == failed.diagnostic.message
    assert terminal[0].cleanup_status is CleanupStatus.CLEANED
