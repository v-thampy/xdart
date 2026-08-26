from __future__ import annotations

from pathlib import Path
from threading import Event, Thread, current_thread
from types import SimpleNamespace

import pytest

from xdart.gui.tabs.scattering.adapters import run_executor as executor_module
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


def test_acquired_sink_is_released_when_session_construction_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Sink:
        def __init__(self, *_args, **_kwargs) -> None:
            self.abort_calls = 0
            self.opened = False

        def abort(self, _result) -> None:
            self.abort_calls += 1
            self.opened = False

    sink = Sink()

    class Opened(_Source):
        def to_scan(self, **_kwargs):
            return _Scan()

    opened = Opened()
    monkeypatch.setattr(executor_module, "open_source", lambda _spec: opened)
    monkeypatch.setattr(executor_module, "load_poni", lambda _path: object())
    monkeypatch.setattr(executor_module, "poni_to_integrator", lambda _poni: object())
    monkeypatch.setattr(executor_module, "build_native_int_reduction_plan_from_args",
                        lambda *_args, **_kwargs: object())
    monkeypatch.setattr(executor_module, "NexusSink", lambda *_args, **_kwargs: sink)

    def failing_session(_plan, _scan, acquired_sink, **_kwargs):
        assert acquired_sink is sink
        sink.opened = True
        raise RuntimeError("session construction failed after sink acquisition")

    monkeypatch.setattr(executor_module, "ScanSession", failing_session)
    run = _run()
    run.configuration = SimpleNamespace(
        thaw_source_spec=lambda: object(), poni_file="calibration.poni", save_path="out.nxs",
        processing_mode="Int 2D", output_mode="Overwrite", mask_file="",
        gi=SimpleNamespace(
            scan_config=lambda: {}, enabled=False, effective_motor="Manual",
            th_val=0.0, tilt_angle=0.0, sample_orientation=1,
        ),
        # This row targets cleanup after sink acquisition, so its forged
        # configuration must satisfy the execution boundary's threshold
        # shape and reach the monkeypatched ScanSession.
        threshold=SimpleNamespace(apply_threshold=False, threshold_min=None,
                                  threshold_max=None, mask_saturation=True),
        bai_1d_args={}, bai_2d_args={}, max_cores=1,
        as_provenance=lambda: {"generation": 1, "fingerprint": "f" * 64,
                               "schema_version": 1}, project_root="",
    )
    run.capture = object()
    executor = StandardRunExecutor()
    executor._active = run

    executor._run(run)

    event = executor.drain_events()[-1]
    assert event.kind is StandardEventKind.FAILED
    assert event.cleanup_status is CleanupStatus.CLEANED
    assert opened.close_calls == 1
    assert sink.abort_calls == 1
    assert sink.opened is False


def test_retry_cannot_drop_sink_whose_abort_never_succeeded() -> None:
    class Session:
        def __init__(self) -> None:
            self.finish_calls = 0
            self.stop_calls = 0

        def finish(self, **_kwargs):
            self.finish_calls += 1
            if self.finish_calls == 1:
                raise RuntimeError("finish failed after retaining sink")
            return SimpleNamespace(failed=False, cancelled=False)

        def stop(self) -> None:
            self.stop_calls += 1

    class Sink:
        def __init__(self) -> None:
            self.abort_calls = 0

        def abort(self, _result) -> None:
            self.abort_calls += 1
            raise RuntimeError("sink still open")

    session, sink = Session(), Sink()
    run = _run(source=_Source(), session=session, sink=sink)
    executor = StandardRunExecutor()
    executor._active = run

    first = executor._cleanup(run)
    assert first.cleanup_status is CleanupStatus.CLEANUP_PENDING
    assert run.session is session
    assert run.sink is sink

    second = executor.close(run.identity)
    assert second.cleanup_status is CleanupStatus.CLEANUP_PENDING
    assert run.sink is sink
    assert sink.abort_calls == 2


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


def test_terminal_progress_does_not_regress_to_zero() -> None:
    class FiveFrameScan:
        name = "Standard"
        frames = ()

        def __len__(self) -> int:
            return 5

    class FinishedSession:
        frames_completed = 5

        def start(self) -> None:
            return None

        def finish(self, **_kwargs):
            return SimpleNamespace(failed=False, cancelled=False, n_processed=5)

    run = _run(source=_Source(), session=FinishedSession())
    run.scan = FiveFrameScan()
    executor = StandardRunExecutor()
    executor._active = run

    executor._run(run)

    terminal = executor.drain_events()[-1]
    assert terminal.kind is StandardEventKind.FINISHED
    assert terminal.completed == 5
    assert terminal.total == 5


def test_cleanup_diagnostics_name_each_failed_owner_in_order() -> None:
    class Session:
        def finish(self, **_kwargs):
            raise RuntimeError("same failure text")

    class Sink:
        def abort(self, _result) -> None:
            raise RuntimeError("same failure text")

    class Source:
        def close(self) -> None:
            raise RuntimeError("same failure text")

    run = _run(source=Source(), session=Session(), sink=Sink())
    receipt = StandardRunExecutor()._cleanup(run)

    assert receipt.cleanup_status is CleanupStatus.CLEANUP_PENDING
    assert tuple(failure.operation for failure in receipt.cleanup_failures) == (
        "session.finish", "sink.abort", "source.close",
    )
