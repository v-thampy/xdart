from __future__ import annotations

from pathlib import Path
from threading import Event, Thread, current_thread

import pytest

from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor, _StandardRun
from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xdart.gui.tabs.scattering.events import CleanupStatus, RunIdentity


def _identity() -> RunIdentity:
    return RunIdentity(1, "f" * 64)


class _Source:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class _Scan:
    name = "Standard"
    frames = ()

    def __len__(self) -> int:
        return 0


def _run(*, source=None, session=None, sink=None) -> _StandardRun:
    return _StandardRun(None, _identity(), _Scan(), source, session, None,
                        Path("out.nxs"), sink=sink)


def _writer_begin_failure(monkeypatch, *, hold_abort=False):
    """Inject storage failure after a real sink acquires its writer."""
    from xrd_tools.reduction import NexusSink

    sinks, aborts, held = [], [], [hold_abort]
    real_begin, real_abort = NexusSink.begin, NexusSink.abort

    def begin(owner, *args, **kwargs):
        real_begin(owner, *args, **kwargs)
        sinks.append(owner)
        raise OSError("writer begin failed after sink acquisition")

    def abort(owner, *args, **kwargs):
        aborts.append(owner)
        if held[0]:
            raise OSError("sink still open")
        return real_abort(owner, *args, **kwargs)

    monkeypatch.setattr(NexusSink, "begin", begin)
    monkeypatch.setattr(NexusSink, "abort", abort)
    return sinks, aborts, held


def test_acquired_sink_is_released_when_session_construction_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    from xrd_tools.sources.image import TiffSeriesSource
    from tests.xdart.scattering.test_e1b1_terminal_cleanup import _prepared_run

    executor, run, _admission = _prepared_run(tmp_path)
    sinks, aborts, _held = _writer_begin_failure(monkeypatch)
    closed = []
    monkeypatch.setattr(TiffSeriesSource, "close", lambda owner: closed.append(owner), raising=False)
    executor._run(run)
    event = executor.drain_events()[-1]
    assert event.kind is StandardEventKind.FAILED
    assert "writer begin failed after sink acquisition" in event.detail
    assert event.cleanup_status is CleanupStatus.CLEANED
    assert len(closed) == 1
    assert len(sinks) == 1 and aborts == sinks
    assert sinks[0]._terminal_result.disposition.value == "aborted"
    assert sinks[0]._writer._h5 is None
    assert run.output is run.session is run.sink is None
    assert executor.close(run.identity).cleanup_status is CleanupStatus.CLEANED


def test_retry_cannot_drop_sink_whose_abort_never_succeeded(tmp_path, monkeypatch) -> None:
    from tests.xdart.scattering.test_e1b1_terminal_cleanup import _prepared_run

    executor, run, admission = _prepared_run(tmp_path)
    sinks, aborts, held = _writer_begin_failure(monkeypatch, hold_abort=True)
    decision = admission.outputs[0]
    try:
        with pytest.raises(OSError, match="writer begin failed after sink acquisition"):
            executor._construct(run, item=decision.item, decision=decision)
        output = run.output
        assert len(sinks) == 1 and output._pending_nexus == sinks
        first = executor._cleanup(run)
        assert first.cleanup_status is CleanupStatus.CLEANUP_PENDING
        assert run.output is output and output._pending_nexus == sinks
        attempts = len(aborts)
        second = executor.close(run.identity)
        assert second.cleanup_status is CleanupStatus.CLEANUP_PENDING
        assert run.output is output and output._pending_nexus == sinks
        assert len(aborts) == attempts + 1
        assert all(owner is sinks[0] for owner in aborts)
    finally:
        held[0] = False
        assert executor.close(run.identity).cleanup_status is CleanupStatus.CLEANED
    assert output._pending_nexus == []
    assert run.output is None


def test_dead_worker_cleanup_retry_never_runs_on_close_caller() -> None:
    calls = []

    class Source:
        def close(self) -> None:
            calls.append(current_thread())
            if len(calls) == 1:
                raise RuntimeError("first close failed")

    run = _run(source=Source())
    executor = StandardRunExecutor()
    executor._active = run
    worker = Thread(target=executor._cleanup, args=(run,), name="scattering-standard")
    run.worker = worker
    worker.start()
    worker.join()

    assert run.cleanup_status is CleanupStatus.CLEANUP_PENDING
    caller = current_thread()
    receipt = executor.close(run.identity)

    assert receipt.cleanup_status is CleanupStatus.CLEANED
    assert len(calls) == 2
    assert calls[1] is not caller


def test_duplicate_close_starts_at_most_one_live_cleanup_retry() -> None:
    entered, release = Event(), Event()
    calls = []

    class Source:
        def close(self) -> None:
            calls.append(current_thread())
            if len(calls) == 1:
                raise RuntimeError("initial cleanup fails")
            if len(calls) == 2:
                entered.set()
                assert release.wait(2)

    run = _run(source=Source())
    executor = StandardRunExecutor(join_timeout=0.01)
    executor._active = run
    first = Thread(target=executor._cleanup, args=(run,), name="initial-cleanup")
    run.worker = first
    first.start()
    first.join()
    try:
        assert executor.close(run.identity).cleanup_status is CleanupStatus.CLEANUP_PENDING
        assert entered.wait(2)
        retry = run.worker
        assert executor.close(run.identity).cleanup_status is CleanupStatus.CLEANUP_PENDING
        assert run.worker is retry
        assert len(calls) == 2
    finally:
        release.set()
        worker = run.worker
        if worker is not None:
            worker.join(2)


def test_terminal_progress_does_not_regress_to_zero(tmp_path) -> None:
    from tests.xdart.scattering.test_e1b1_terminal_cleanup import _prepared_run

    executor, run, _admission = _prepared_run(tmp_path, frame_count=5)
    executor._run(run)
    events = executor.drain_events()
    terminal = events[-1]
    assert terminal.kind is StandardEventKind.FINISHED, terminal.detail
    assert terminal.completed == 5
    assert terminal.total == 5
    frame_events = [event for event in events if event.kind is StandardEventKind.FRAME_READY]
    assert frame_events and terminal.completed >= max(event.completed for event in frame_events)
    assert executor.close(run.identity).cleanup_status is CleanupStatus.CLEANED


def test_cleanup_diagnostics_name_each_failed_owner_in_order(tmp_path, monkeypatch) -> None:
    from xrd_tools.sources.image import TiffSeriesSource
    from tests.xdart.scattering.test_e1b1_terminal_cleanup import _prepared_run

    executor, run, admission = _prepared_run(tmp_path)
    _sinks, _aborts, held = _writer_begin_failure(monkeypatch, hold_abort=True)
    decision = admission.outputs[0]
    def close_source(owner):
        raise RuntimeError("sink still open")
    monkeypatch.setattr(TiffSeriesSource, "close", close_source, raising=False)
    try:
        with pytest.raises(OSError, match="writer begin failed after sink acquisition"):
            executor._construct(run, item=decision.item, decision=decision)
        receipt = executor._cleanup(run)
        assert receipt.cleanup_status is CleanupStatus.CLEANUP_PENDING
        assert tuple(failure.operation for failure in receipt.cleanup_failures) == (
            "dynamic_output.finish", "source.close",
        )
        assert tuple(failure.message for failure in receipt.cleanup_failures) == (
            "sink still open", "sink still open",
        )
    finally:
        held[0] = False
        monkeypatch.setattr(TiffSeriesSource, "close", lambda owner: None)
        assert executor.close(run.identity).cleanup_status is CleanupStatus.CLEANED
