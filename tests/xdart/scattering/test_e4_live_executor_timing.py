from __future__ import annotations

import logging
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from xrd_tools.session.run_configuration import RunIntent
from xdart.gui.tabs.scattering.adapters import run_executor as executor_module
from xdart.gui.tabs.scattering.adapters.run_executor import (
    StandardRunExecutor,
    _StandardRun,
)
from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xdart.gui.tabs.scattering.events import (
    CleanupStatus,
    ExecutorClosed,
    RunIdentity,
)


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


def test_display_projection_leaves_hdf5_completion_thread(
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
        Path("display-worker.nxs"),
    )
    executor = StandardRunExecutor()
    entered = threading.Event()
    release = threading.Event()
    worker_names: list[str] = []

    def project(_run, _event, _image) -> None:
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
