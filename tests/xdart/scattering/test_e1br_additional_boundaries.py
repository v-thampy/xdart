from __future__ import annotations

from pathlib import Path
from threading import Barrier, Event, Thread, current_thread, enumerate as live_threads
from types import SimpleNamespace

import numpy as np
import pytest

from xdart.gui.tabs.scattering.adapters.run_executor import (
    StandardRunExecutor,
    _StandardRun,
)
from xdart.gui.tabs.scattering.display_values import (
    DisplayFrameKey,
    StandardDisplayPayload,
    StandardEventKind,
    display_payload_is_valid,
)
from xdart.gui.tabs.scattering.events import (
    CleanupStatus,
    RunIdentity,
)
from xrd_tools.core import Axis, FrameView


def _identity() -> RunIdentity:
    return RunIdentity(1, "f" * 64)


def _frame(identity: RunIdentity) -> DisplayFrameKey:
    return DisplayFrameKey(identity, "scan", "artifact.nxs", 1, 1)


def test_primary_finish_failure_requires_explicit_sink_release() -> None:
    class Scan:
        name = "Standard"
        frames = ()

        def __len__(self) -> int:
            return 0

    class Session:
        frames_completed = 0

        def __init__(self) -> None:
            self.finish_calls = 0

        def start(self) -> None:
            return None

        def finish(self, **_kwargs):
            self.finish_calls += 1
            if self.finish_calls == 1:
                raise RuntimeError("primary finish failed after retaining sink")
            return SimpleNamespace(failed=False, cancelled=False)

    class Sink:
        def __init__(self) -> None:
            self.abort_calls = 0

        def abort(self, _result) -> None:
            self.abort_calls += 1

    class Source:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    identity = _identity()
    session, sink, source = Session(), Sink(), Source()
    run = _StandardRun(
        None,
        identity,
        Scan(),
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
    assert session.finish_calls == 2
    assert source.close_calls == 1
    assert (
        sink.abort_calls == 1
        or terminal.cleanup_status is CleanupStatus.CLEANUP_PENDING
    ), "a retrying finish cannot by itself prove that the acquired sink released"


def test_display_validator_rejects_throwing_nested_axis_before_render() -> None:
    class ThrowingAxis:
        @property
        def values(self):
            raise RuntimeError("untrusted axis")

    identity = _identity()
    view = FrameView(
        label=1,
        axis_1d=Axis("Q", "q_A^-1", values=np.array([1.0, 2.0])),
        intensity_1d=np.array([3.0, 4.0]),
    )
    object.__setattr__(view, "axis_1d", ThrowingAxis())
    frame = _frame(identity)
    payload = StandardDisplayPayload(2, frame, "bad axis", view)

    assert display_payload_is_valid(payload, identity, frame, 2) is False


def test_mismatched_1d_axis_is_rejected_before_shell_projection() -> None:
    identity = _identity()
    view = FrameView(
        label=1,
        axis_1d=Axis(
            "Q",
            "q_A^-1",
            values=np.array([1.0, 2.0, 3.0]),
        ),
        intensity_1d=np.array([4.0, 5.0, 6.0]),
    )
    object.__setattr__(
        view, "intensity_1d", np.array([4.0, 5.0])
    )
    frame = _frame(identity)
    payload = StandardDisplayPayload(1, frame, "malformed 1d", view)

    assert display_payload_is_valid(payload, identity, frame, 1) is False


@pytest.mark.parametrize("bad_field", ["frame", "generation"])
def test_boolean_frame_or_generation_is_rejected_before_shell_projection(
    bad_field: str,
) -> None:
    identity = _identity()
    frame = _frame(identity)
    payload_frame: object = True if bad_field == "frame" else frame
    payload_generation: object = True if bad_field == "generation" else 1
    payload = StandardDisplayPayload(
        payload_generation,  # type: ignore[arg-type]
        payload_frame,  # type: ignore[arg-type]
        "non-exact scalar",
        FrameView(1, raw=np.ones((2, 3))),
    )

    assert display_payload_is_valid(payload, identity, frame, 1) is False


def test_empty_2d_axes_are_rejected_before_shell_projection() -> None:
    identity = _identity()
    view = FrameView(
        label=1,
        axis_2d_x=Axis(
            "Q", "q_A^-1", values=np.array([], dtype=float)
        ),
        axis_2d_y=Axis(
            "chi", "chi_deg", values=np.array([], dtype=float)
        ),
        intensity_2d=np.empty((0, 0), dtype=float),
    )
    frame = _frame(identity)
    payload = StandardDisplayPayload(1, frame, "empty cake", view)

    assert display_payload_is_valid(payload, identity, frame, 1) is False


def test_mismatched_2d_axes_are_rejected_before_shell_projection() -> None:
    identity = _identity()
    view = FrameView(
        label=1,
        axis_2d_x=Axis(
            "Q", "q_A^-1", values=np.array([1.0, 2.0, 3.0])
        ),
        axis_2d_y=Axis(
            "chi", "chi_deg", values=np.array([-5.0, 5.0])
        ),
        intensity_2d=np.ones((2, 3)),
    )
    object.__setattr__(
        view,
        "axis_2d_x",
        Axis("Q", "q_A^-1", values=np.array([1.0, 2.0])),
    )
    frame = _frame(identity)
    payload = StandardDisplayPayload(1, frame, "mismatched cake", view)

    assert display_payload_is_valid(payload, identity, frame, 1) is False


def test_concurrent_close_starts_only_one_live_cleanup_retry() -> None:
    entered, release = Event(), Event()
    callers = Barrier(3)
    calls = []

    class Source:
        def close(self) -> None:
            calls.append(current_thread())
            if len(calls) == 1:
                raise RuntimeError("initial cleanup fails")
            entered.set()
            assert release.wait(2)

    identity = _identity()
    run = _StandardRun(
        None, identity, None, Source(), None, None,
        Path("unused.nxs"),
    )
    executor = StandardRunExecutor(join_timeout=0.01)
    executor._active = run
    initial = Thread(target=executor._cleanup, args=(run,), name="initial-cleanup")
    run.worker = initial
    initial.start()
    initial.join(2)
    receipts = []

    def close() -> None:
        callers.wait()
        receipts.append(executor.close(identity))

    left = Thread(target=close, name="left-close")
    right = Thread(target=close, name="right-close")
    left.start()
    right.start()
    callers.wait()
    try:
        left.join(2)
        right.join(2)
        assert entered.wait(2)
        assert len(calls) == 2
        assert len([
            thread for thread in live_threads()
            if thread.name == "scattering-cleanup" and thread.is_alive()
        ]) == 1
        assert len(receipts) == 2
        assert all(
            receipt.cleanup_status is CleanupStatus.CLEANUP_PENDING
            for receipt in receipts
        )
    finally:
        release.set()
        worker = run.worker
        if worker is not None:
            worker.join(2)
