"""Exact-tip E1b.2 probe using the real ScanSession idempotence contract."""

from __future__ import annotations

from pathlib import Path

from xrd_tools.reduction import ReductionPlan, Scan
from xrd_tools.session.scan_session import ScanSession
from xdart.gui.tabs.scattering.adapters.run_executor import (
    StandardRunExecutor,
    _StandardRun,
)
from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xdart.gui.tabs.scattering.events import CleanupStatus, RunIdentity


class _FinishFailureSink:
    def __init__(self) -> None:
        self.begin_calls = 0
        self.finish_calls = 0
        self.abort_calls = 0
        self.live = False

    def begin(self, _scan, _plan) -> None:
        self.begin_calls += 1
        self.live = True

    def finish(self, _result) -> None:
        self.finish_calls += 1
        raise RuntimeError("sink finish left the sink live")

    def abort(self, _result) -> None:
        self.abort_calls += 1
        raise RuntimeError("sink abort still failed")


class _RecoverableFinishFailureSink(_FinishFailureSink):
    def abort(self, _result) -> None:
        self.abort_calls += 1
        self.live = False


class _Source:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def test_real_scansession_finish_failure_cannot_become_cleaned() -> None:
    sink = _FinishFailureSink()
    scan = Scan("empty", [], integrator=object())
    session = ScanSession(
        ReductionPlan(integration_2d=None),
        scan,
        sink=sink,
        executor=1,
    )
    source = _Source()
    identity = RunIdentity(1, "f" * 64)
    run = _StandardRun(
        None,
        identity,
        scan,
        source,
        session,
        None,
        Path("unused.nxs"),
        sink=sink,
    )
    executor = StandardRunExecutor()
    executor._active = run

    executor._run(run)
    terminal = executor.drain_events()[-1]

    assert terminal.kind is StandardEventKind.FAILED
    assert terminal.cleanup_status is CleanupStatus.CLEANUP_PENDING
    assert run.sink is sink
    assert sink.live is True
    assert sink.finish_calls == 1
    assert sink.abort_calls >= 1
    assert source.close_calls == 1


def test_real_scansession_finish_failure_requires_successful_abort_to_clean() -> None:
    sink = _RecoverableFinishFailureSink()
    scan = Scan("empty", [], integrator=object())
    session = ScanSession(
        ReductionPlan(integration_2d=None),
        scan,
        sink=sink,
        executor=1,
    )
    source = _Source()
    identity = RunIdentity(1, "f" * 64)
    run = _StandardRun(
        None,
        identity,
        scan,
        source,
        session,
        None,
        Path("unused.nxs"),
        sink=sink,
    )
    executor = StandardRunExecutor()
    executor._active = run

    executor._run(run)
    terminal = executor.drain_events()[-1]

    assert terminal.kind is StandardEventKind.FAILED
    assert terminal.cleanup_status is CleanupStatus.CLEANED
    assert run.sink is None
    assert sink.live is False
    assert sink.finish_calls == 1
    assert sink.abort_calls == 1
    assert source.close_calls == 1
