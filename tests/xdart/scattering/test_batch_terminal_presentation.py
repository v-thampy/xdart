from __future__ import annotations

from xdart.gui.tabs.scattering.batch_terminal_presentation import (
    CONSUME,
    FULL_READY,
    FULL_REFUSED,
    FULL_REQUESTED,
    HOLD,
    PAINT,
    QUALIFY,
    RETAIN,
    RETIRE,
    SELECT,
    BatchTerminalPresentationController,
)
from xdart.gui.tabs.scattering.display_values import (
    DisplayFrameKey,
    StandardEventKind,
    StandardRunEvent,
)
from xdart.gui.tabs.scattering.events import CleanupStatus, RunIdentity
from xdart.gui.tabs.scattering.shell_values import ProgressProjection


def _frame(identity: RunIdentity, ordinal: int) -> DisplayFrameKey:
    return DisplayFrameKey(identity, "scan", "/out/result.nxs", ordinal, ordinal)


def _frame_event(frame: DisplayFrameKey, total: int = 3) -> StandardRunEvent:
    return StandardRunEvent(
        frame.run_identity,
        StandardEventKind.FRAME_READY,
        completed=frame.work_ordinal,
        total=total,
        artifact=frame.artifact,
        frame_key=frame,
    )


def _terminal(
    identity: RunIdentity,
    *,
    completed: int = 3,
    kind: StandardEventKind = StandardEventKind.FINISHED,
    cleanup: CleanupStatus = CleanupStatus.CLEANED,
) -> StandardRunEvent:
    return StandardRunEvent(
        identity,
        kind,
        completed=completed,
        total=3,
        artifact="/out/result.nxs",
        cleanup_status=cleanup,
    )


def _facts(
    identity: RunIdentity,
    frame: DisplayFrameKey,
    *,
    generation: int | None = None,
) -> tuple[object, ...]:
    return identity, frame, frame, (frame,), True, generation


def test_batch_frames_are_held_and_only_exact_clean_latest_can_paint() -> None:
    identity = RunIdentity(1, "batch")
    controller = BatchTerminalPresentationController()
    frozen = ProgressProjection(detail="Run started")
    controller.begin_run(identity, batch_mode=True, visible_progress=frozen)
    frames = tuple(_frame(identity, ordinal) for ordinal in (1, 2, 3))

    for frame in frames:
        assert controller.record_frame(_frame_event(frame), frame)
        assert controller.latest_frame is frame
        assert controller.project_progress(
            ProgressProjection(frame.work_ordinal, 3, "worker")
        ) is frozen

    decision = controller.begin_terminal(
        _terminal(identity), _facts(identity, frames[-1]),
    )
    assert decision.action == SELECT and decision.frame is frames[-1]
    decision = controller.complete_terminal(
        frames[-1],
        _facts(identity, frames[-1]),
        full_raw=FULL_READY,
    )
    assert decision.action == PAINT
    owner = decision.presentation
    assert owner is not None and owner.frame is frames[-1]
    assert controller.ready_to_paint(
        _facts(identity, frames[-1])
    ) is owner

    assert controller.retire().action == HOLD
    assert controller.complete_paint(owner, applied=False) is False
    assert controller.active and controller.presentation is owner
    assert controller.complete_paint(owner, applied=True)
    assert not controller.active
    assert controller.presentation is not owner
    assert controller.presentation is not None and controller.presentation.painted


def test_full_raw_wait_is_exact_and_duplicate_display_is_consumed() -> None:
    identity = RunIdentity(2, "full")
    frame = _frame(identity, 3)
    controller = BatchTerminalPresentationController()
    controller.begin_run(
        identity, batch_mode=True, visible_progress=ProgressProjection(),
    )
    controller.record_frame(_frame_event(frame), frame)
    assert controller.begin_terminal(
        _terminal(identity), _facts(identity, frame),
    ).action == SELECT
    waiting = controller.complete_terminal(
        frame,
        _facts(identity, frame),
        full_raw=FULL_REQUESTED,
    ).presentation
    assert waiting is not None and waiting.awaiting_full_raw

    ready = StandardRunEvent(
        identity,
        StandardEventKind.DISPLAY_READY,
        artifact=frame.artifact,
        frame_key=frame,
        selection_generation=7,
    )
    decision = controller.inspect_display_event(
        ready, _facts(identity, frame, generation=7),
    )
    assert decision.action == QUALIFY and decision.presentation is waiting
    assert not controller.accept_qualified_display(waiting, ready, _frame(identity, 3))
    assert controller.accept_qualified_display(waiting, ready, frame)
    paintable = controller.presentation
    assert paintable is not None and not paintable.awaiting_full_raw
    assert controller.complete_paint(paintable, applied=True)

    painted = controller.presentation
    duplicate = controller.inspect_display_event(
        ready, _facts(identity, frame, generation=7),
    )
    assert duplicate.action == CONSUME and duplicate.presentation is painted


def test_refusals_keep_a_fail_closed_terminal_fence() -> None:
    identity = RunIdentity(3, "refuse")
    frame = _frame(identity, 3)
    controller = BatchTerminalPresentationController()
    controller.begin_run(
        identity, batch_mode=True, visible_progress=ProgressProjection(),
    )
    controller.record_frame(_frame_event(frame), frame)
    refusal = controller.begin_terminal(
        _terminal(identity, kind=StandardEventKind.FAILED),
        _facts(identity, frame),
    )
    assert refusal.action == RETAIN
    assert refusal.presentation is not None and refusal.presentation.frame is None
    assert controller.active and controller.retire().action == HOLD

    replacement = RunIdentity(4, "replacement")
    retired = controller.begin_run(
        replacement, batch_mode=True, visible_progress=ProgressProjection(),
    )
    assert retired.action == RETIRE
    assert controller.active and controller.presentation is None


def test_full_raw_refusal_retires_only_on_explicit_close_or_new_run() -> None:
    identity = RunIdentity(5, "full-refused")
    frame = _frame(identity, 3)
    controller = BatchTerminalPresentationController()
    controller.begin_run(
        identity, batch_mode=True, visible_progress=ProgressProjection(),
    )
    controller.record_frame(_frame_event(frame), frame)
    controller.begin_terminal(
        _terminal(identity), _facts(identity, frame),
    )
    refusal = controller.complete_terminal(
        frame,
        _facts(identity, frame),
        full_raw=FULL_REFUSED,
        diagnostic="Full Raw refused; prior display retained.",
    )
    assert refusal.action == RETAIN and refusal.presentation is not None
    assert refusal.presentation.frame is None and controller.active
    assert controller.retire().action == HOLD
    assert controller.retire(force=True).action == RETIRE
    assert not controller.active and controller.presentation is None
