"""Qt-free ownership of Batch terminal presentation state and transitions."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .display_values import DisplayFrameKey, StandardEventKind, StandardRunEvent
from .events import CleanupStatus, RunIdentity
from .shell_values import ProgressProjection


HOLD, SELECT, PAINT, QUALIFY, CONSUME = "hold", "select", "paint", "qualify", "consume"
PASS_THROUGH, RETAIN, RETIRE, IDLE = (
    "pass_through", "retain", "retire", "idle",
)
FULL_READY, FULL_REQUESTED, FULL_REFUSED = (
    "ready", "requested", "refused",
)
_OWNER_CHANGED = "Batch terminal owner changed; prior display retained."
_UNCLEAN = "Batch did not finish cleanly; prior display retained."
_NOT_EXACT = "Batch terminal frame was not exact; prior display retained."
_RAW_REFUSED = "Batch terminal Full Raw request was refused; prior display retained."
BatchNavigationFacts = tuple[
    RunIdentity | None, DisplayFrameKey | None, DisplayFrameKey | None,
    tuple[DisplayFrameKey, ...], bool, int | None,
]


@dataclass(frozen=True, slots=True)
class BatchTerminalPresentation:
    """Immutable receipt for one exact Batch terminal presentation."""

    run_identity: RunIdentity
    frame: DisplayFrameKey | None
    awaiting_full_raw: bool = False
    painted: bool = False

    def __post_init__(self) -> None:
        frame = self.frame
        if (
            type(self.run_identity) is not RunIdentity
            or frame is not None and (
                type(frame) is not DisplayFrameKey
                or frame.run_identity is not self.run_identity
            )
            or type(self.awaiting_full_raw) is not bool
            or type(self.painted) is not bool
            or frame is None and (self.awaiting_full_raw or self.painted)
            or self.awaiting_full_raw and self.painted
        ):
            raise TypeError("Batch terminal presentation is invalid")


@dataclass(frozen=True, slots=True)
class BatchTerminalDecision:
    """One explicit action/effect returned to the Qt/context adapter."""

    action: str
    presentation: BatchTerminalPresentation | None = None
    frame: DisplayFrameKey | None = None
    notice: str = ""


def _navigation_matches(
    owner: BatchTerminalPresentation, facts: BatchNavigationFacts,
) -> bool:
    context_identity, _, current, selected, owns_frame, _ = facts
    frame = owner.frame
    return (
        frame is not None
        and context_identity is owner.run_identity
        and current is frame
        and selected == (frame,)
        and owns_frame
    )


class BatchTerminalPresentationController:
    """Own Batch suppression, exact terminal custody, and retirement."""

    def __init__(self) -> None:
        self._active_run: RunIdentity | None = None
        self._latest_frame: DisplayFrameKey | None = None
        self._visible_progress: ProgressProjection | None = None
        self._presentation: BatchTerminalPresentation | None = None

    active = property(lambda self: self._active_run is not None)
    latest_frame = property(lambda self: self._latest_frame)
    presentation = property(lambda self: self._presentation)
    needs_polling = property(lambda self: bool(
        self._presentation and self._presentation.frame and not self._presentation.painted
    ))

    def begin_run(self, run_identity: RunIdentity, *, batch_mode: bool,
                  visible_progress: ProgressProjection) -> BatchTerminalDecision:
        retirement = self.retire(force=True)
        if batch_mode:
            self._active_run = run_identity
            self._visible_progress = visible_progress
        return retirement

    def record_frame(self, event: StandardRunEvent,
                     frame: DisplayFrameKey) -> bool:
        identity = self._active_run
        if identity is None:
            return False
        if (
            event.kind is StandardEventKind.FRAME_READY
            and event.run_identity is identity
            and event.frame_key is frame
            and frame.run_identity is identity
        ):
            self._latest_frame = frame
        return True

    def record_stop(self, progress: ProgressProjection) -> None:
        if self.active: self._visible_progress = progress

    def project_progress(self, current: ProgressProjection) -> ProgressProjection:
        return self._visible_progress if self.active and self._visible_progress is not None else current

    def begin_terminal(self, event: StandardRunEvent,
                       facts: BatchNavigationFacts) -> BatchTerminalDecision:
        context_identity, navigation_last, _, _, owns_latest, _ = facts
        identity = self._active_run
        if identity is None:
            return BatchTerminalDecision(IDLE)
        self._visible_progress = None
        owner = self._presentation
        if event.run_identity is not identity:
            return self._reject(identity, _OWNER_CHANGED)
        if owner is not None:
            if owner.run_identity is not identity:
                return self._reject(identity, _OWNER_CHANGED)
            return BatchTerminalDecision(RETAIN, presentation=owner)
        if (
            event.kind is not StandardEventKind.FINISHED
            or event.cleanup_status is not CleanupStatus.CLEANED
        ):
            return self._reject(identity, _UNCLEAN)
        latest = self._latest_frame
        if not (
            latest is not None
            and latest.run_identity is identity
            and 0 < event.completed <= event.total
            and latest.work_ordinal == event.completed
            and latest.artifact == event.artifact
            and context_identity is identity
            and navigation_last is latest
            and owns_latest
        ):
            return self._reject(identity, _NOT_EXACT)
        return BatchTerminalDecision(SELECT, frame=latest)

    def complete_terminal(
        self, frame: DisplayFrameKey, facts: BatchNavigationFacts, *,
        full_raw: str, diagnostic: str = "",
    ) -> BatchTerminalDecision:
        context_identity, _, current, selected, owns_frame, _ = facts
        identity = self._active_run
        if not (
            identity is frame.run_identity
            and self._presentation is None
            and self._latest_frame is frame
            and context_identity is identity
            and current is frame
            and selected == (frame,)
            and owns_frame
        ):
            return self._reject(identity or frame.run_identity, _NOT_EXACT)
        if full_raw == FULL_REFUSED:
            return self._reject(identity, diagnostic or _RAW_REFUSED)
        waiting = full_raw == FULL_REQUESTED
        owner = BatchTerminalPresentation(identity, frame, awaiting_full_raw=waiting)
        self._presentation = owner
        return BatchTerminalDecision(HOLD if waiting else PAINT, owner, frame)

    def inspect_display_event(self, event: StandardRunEvent,
                              facts: BatchNavigationFacts) -> BatchTerminalDecision:
        owner = self._presentation
        exact = bool(
            owner is not None
            and _navigation_matches(owner, facts)
            and event.run_identity is owner.run_identity
            and event.frame_key is owner.frame
        )
        if (
            not self.active and exact and owner is not None and owner.painted
            and event.artifact == owner.frame.artifact
            and facts[-1] is not None
            and event.selection_generation == facts[-1]
        ):
            return BatchTerminalDecision(CONSUME, presentation=owner)
        if not self.active:
            return BatchTerminalDecision(PASS_THROUGH)
        if exact and owner is not None and owner.awaiting_full_raw and not owner.painted:
            return BatchTerminalDecision(QUALIFY, presentation=owner)
        return BatchTerminalDecision(HOLD, presentation=owner)

    def accept_qualified_display(
        self, owner: BatchTerminalPresentation, event: StandardRunEvent,
        payload_frame: DisplayFrameKey | None,
    ) -> bool:
        if not (
            self._presentation is owner
            and self._active_run is owner.run_identity
            and owner.awaiting_full_raw
            and not owner.painted
            and owner.frame is not None
            and event.run_identity is owner.run_identity
            and event.frame_key is owner.frame
            and payload_frame is owner.frame
        ):
            return False
        self._presentation = replace(owner, awaiting_full_raw=False)
        return True

    def ready_to_paint(
        self, facts: BatchNavigationFacts,
    ) -> BatchTerminalPresentation | None:
        owner = self._presentation
        return owner if (
            owner is not None
            and self._active_run is owner.run_identity
            and not owner.awaiting_full_raw
            and not owner.painted
            and _navigation_matches(owner, facts)
        ) else None

    def complete_paint(self, owner: BatchTerminalPresentation, *, applied: bool) -> bool:
        if not applied or self._presentation is not owner:
            return False
        self._presentation = replace(owner, painted=True)
        self._active_run = self._latest_frame = self._visible_progress = None
        return True

    def retire(self, *, force: bool = False) -> BatchTerminalDecision:
        owner = self._presentation
        if not any((self._active_run, self._latest_frame, self._visible_progress, owner)):
            return BatchTerminalDecision(IDLE)
        if not force and (owner is None or not owner.painted):
            return BatchTerminalDecision(HOLD, presentation=owner)
        decision = BatchTerminalDecision(RETIRE, presentation=owner)
        self._active_run = self._latest_frame = self._visible_progress = None
        self._presentation = None
        return decision

    def _reject(self, identity: RunIdentity, notice: str) -> BatchTerminalDecision:
        owner = BatchTerminalPresentation(identity, None)
        self._presentation, self._latest_frame = owner, None
        return BatchTerminalDecision(RETAIN, presentation=owner, notice=notice)
