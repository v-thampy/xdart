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
from xdart.gui.tabs.scattering.display_values import (
    StandardEventKind,
    StandardRunEvent,
)
from xdart.gui.tabs.scattering.events import (
    CleanupStatus,
    ExecutorClosed,
    RunIdentity,
)
from xdart.gui.tabs.scattering.page import ScatteringWorkspace


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
    configuration = RunIntent().freeze()
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
        resources=SimpleNamespace(
            admission=receipt,
            directory_session=None,
        ),
    )
    executor = StandardRunExecutor()
    constructs = 0

    def construct(current, *, item, labels, decision) -> None:
        nonlocal constructs
        constructs += 1
        assert current.artifact == item.target
        if constructs == 2:
            assert current.current_total == 2
            assert current.current_completed == 0
            assert current.current_published == 0
            raise RuntimeError("second construct failed")

    def execute_current(current, *, construct=False) -> bool:
        assert construct is False
        current.completed += current.current_total
        current.current_completed = current.current_total
        current.current_published = current.current_total
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
        "target_state_matches",
        lambda _decision: True,
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
