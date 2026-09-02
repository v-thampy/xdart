from __future__ import annotations

from pathlib import Path

import pytest

from xdart.gui.tabs.scattering.adapters import run_executor as module
from xdart.gui.tabs.scattering.adapters.run_executor import (
    StandardRunExecutor,
    _StandardRun,
)
from xdart.gui.tabs.scattering.events import CleanupStatus, RunIdentity


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _Display:
    def __init__(self, clock: _Clock, waits) -> None:
        self._clock = clock
        self._waits = list(waits)
        self.received: list[float] = []

    def retire(self, *, join_timeout: float) -> bool:
        self.received.append(join_timeout)
        duration, result = self._waits.pop(0)
        self._clock.advance(min(duration, join_timeout))
        return result


class _Worker:
    def __init__(self, clock: _Clock, duration: float, *, dies: bool) -> None:
        self._clock = clock
        self._duration = duration
        self._dies = dies
        self._alive = True
        self.received: list[float] = []

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout: float | None = None) -> None:
        assert timeout is not None
        self.received.append(timeout)
        elapsed = min(self._duration, timeout)
        self._clock.advance(elapsed)
        if self._dies and elapsed >= self._duration:
            self._alive = False


class _ContextRuntime:
    def __init__(self, order: list[str]) -> None:
        self._order = order

    def retire(self) -> None:
        self._order.append("context")


def _run(display: _Display, worker: _Worker) -> _StandardRun:
    identity = RunIdentity(1, "f" * 64)
    run = _StandardRun(
        None, identity, None, None, None, None, Path("out.nexus")
    )
    run.display = display
    run.worker = worker
    return run


def test_close_run_shares_one_deadline_across_all_four_waits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    order: list[str] = []
    display = _Display(clock, ((1.0, False), (1.0, False)))
    original = _Worker(clock, 1.0, dies=True)
    cleanup = _Worker(clock, 1.0, dies=True)
    run = _run(display, original)
    run.context_runtime = _ContextRuntime(order)
    executor = StandardRunExecutor(join_timeout=5.0)
    executor._active = run

    def stop(identity: RunIdentity) -> None:
        assert identity is run.identity
        order.append("stop")
        clock.advance(1.0)

    monkeypatch.setattr(module, "monotonic", clock)
    monkeypatch.setattr(executor, "stop", stop)
    monkeypatch.setattr(
        executor, "_start_cleanup_retry", lambda exact: cleanup,
    )

    receipt = executor._close_run(run.identity)

    assert receipt.cleanup_status is CleanupStatus.CLEANUP_PENDING
    assert order == ["stop", "context"]
    assert display.received == pytest.approx([4.0, 2.0])
    assert original.received == pytest.approx([3.0])
    assert cleanup.received == pytest.approx([1.0])
    assert clock.now == pytest.approx(5.0)


def test_close_run_exhausted_deadline_still_attempts_zero_time_convergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    display = _Display(clock, ((2.0, False), (1.0, False)))
    original = _Worker(clock, 3.0, dies=True)
    cleanup = _Worker(clock, 1.0, dies=True)
    run = _run(display, original)
    executor = StandardRunExecutor(join_timeout=5.0)
    executor._active = run
    monkeypatch.setattr(module, "monotonic", clock)
    monkeypatch.setattr(executor, "stop", lambda _identity: None)
    monkeypatch.setattr(
        executor, "_start_cleanup_retry", lambda exact: cleanup,
    )

    receipt = executor._close_run(run.identity)

    assert receipt.cleanup_status is CleanupStatus.CLEANUP_PENDING
    assert display.received == pytest.approx([5.0, 0.0])
    assert original.received == pytest.approx([3.0])
    assert cleanup.received == pytest.approx([0.0])
    assert clock.now == pytest.approx(5.0)


def test_close_run_live_original_worker_stops_follow_on_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    display = _Display(clock, ((1.0, False),))
    original = _Worker(clock, 10.0, dies=False)
    run = _run(display, original)
    executor = StandardRunExecutor(join_timeout=5.0)
    executor._active = run
    cleanup_calls = []
    monkeypatch.setattr(module, "monotonic", clock)
    monkeypatch.setattr(executor, "stop", lambda _identity: None)
    monkeypatch.setattr(
        executor,
        "_start_cleanup_retry",
        lambda exact: cleanup_calls.append(exact),
    )

    receipt = executor._close_run(run.identity)

    assert receipt.cleanup_status is CleanupStatus.CLEANUP_PENDING
    assert display.received == pytest.approx([5.0])
    assert original.received == pytest.approx([4.0])
    assert cleanup_calls == []
    assert clock.now == pytest.approx(5.0)
