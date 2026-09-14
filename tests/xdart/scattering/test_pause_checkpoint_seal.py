"""A durable Pause seals the quiesced writer's checkpoint.

Every record write revokes an artifact's checkpoint-hydration authority and
only a seal re-authorises it, so a Pause that lands between the sixteen-frame
seals would otherwise leave every evicted frame unreadable for as long as it
lasts.  The runtime therefore flushes the paused session right after its
writer drains, under the same command lock, and before the display projection
drains — and never when the drain did not happen.
"""

from __future__ import annotations

import pytest

from xdart.gui.tabs.scattering.acquisition_runtime import (
    AcquisitionRuntime,
    CommandCompensationFailure,
)
from xdart.gui.tabs.scattering.events import DurablePaused, RunIdentity


class _Session:
    def __init__(self, *, drained: bool = True) -> None:
        self.drained = drained
        self.log: list[str] = []
        self.flush_error: BaseException | None = None
        self.resume_error: BaseException | None = None

    def pause(self, *, timeout: float) -> bool:
        self.log.append("pause")
        return self.drained

    def flush(self, *, force: bool) -> None:
        self.log.append(f"flush(force={force})")
        if self.flush_error is not None:
            raise self.flush_error

    def resume(self) -> None:
        self.log.append("resume")
        if self.resume_error is not None:
            raise self.resume_error


@pytest.mark.parametrize("live", (False, True))
def test_durable_pause_seals_after_drain_and_before_projection(live: bool):
    runtime, session = AcquisitionRuntime(), _Session()
    identity = RunIdentity(1, "seal")
    if live:
        runtime._arm_live()

    def drain_projection(_timeout: float) -> bool:
        session.log.append("projection")
        return True

    assert runtime.pause(
        session,
        identity,
        1.0,
        drain_projection=drain_projection,
        session_supplier=lambda: session,
    ) == DurablePaused(identity, 1)
    assert session.log == ["pause", "flush(force=True)", "projection"]


def test_pause_that_did_not_drain_seals_nothing():
    runtime, session = AcquisitionRuntime(), _Session(drained=False)
    with pytest.raises(TimeoutError, match="acquisition did not reach"):
        runtime.pause(session, RunIdentity(1, "timeout"), 0.01)
    assert session.log == ["pause", "resume"]
    assert runtime._gate.is_set()


def test_live_pause_without_a_session_seals_nothing():
    runtime = AcquisitionRuntime()
    runtime._arm_live()
    identity = RunIdentity(1, "between-partitions")
    assert runtime.pause(
        None, identity, 1.0, session_supplier=lambda: None,
    ) == DurablePaused(identity, 1)


def test_seal_failure_compensates_to_running_and_raises_the_seal_error():
    runtime, session = AcquisitionRuntime(), _Session()
    session.flush_error = RuntimeError("writer flush incomplete: sealed rows")
    projected: list[float] = []
    with pytest.raises(RuntimeError, match="writer flush incomplete"):
        runtime.pause(
            session,
            RunIdentity(1, "seal-failed"),
            1.0,
            drain_projection=lambda timeout: not projected.append(timeout),
        )
    assert session.log == ["pause", "flush(force=True)", "resume"]
    assert projected == []
    assert runtime._gate.is_set()


def test_seal_failure_with_failed_compensation_keeps_both_exact_causes():
    runtime, session = AcquisitionRuntime(), _Session()
    session.flush_error = RuntimeError("seal failed")
    session.resume_error = RuntimeError("resume after seal failed")
    with pytest.raises(CommandCompensationFailure) as captured:
        runtime.pause(session, RunIdentity(1, "seal-compensation"), 1.0)
    assert tuple(item.message for item in captured.value.diagnostics) == (
        "seal failed",
        "resume after seal failed",
    )
    assert tuple(item.operation for item in captured.value.diagnostics) == (
        "context.pause",
        "context.pause.compensation",
    )
    assert session.log == ["pause", "flush(force=True)", "resume"]
    assert runtime._gate.is_set() is False
