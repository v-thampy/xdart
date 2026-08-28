"""Qt-free ownership for workspace experiment operations.

The page composes notices, dialogs, and Browse mutations.  This owner keeps
the one experiment worker slot and active operation state.  Terminal reload
directives transfer immediately to ``ProcessedBrowserOwner`` through typed
transitions; this owner does not retain a second copy.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import os
from typing import Mapping

from xrd_tools.io.output_transaction import (
    StreamTerminal,
    stream_terminal_object_revision,
)
from xrd_tools.reduction import ReintegrateResult
from xrd_tools.session.run_configuration import FrozenRunConfiguration

from .adapters.external_operation import OperationSlot
from .browse_values import LoadedBrowseCapture
from .events import CleanupStatus
from .operation_values import (
    OperationCleanupReceipt,
    OperationContextStamp,
    OperationIdentity,
    OperationPending,
    OperationTerminalStatus,
    OperationUpdate,
)
from .processed_browser import (
    AverageReloadDirective,
    ReintegrateReloadDirective,
)


class WorkspaceRefreshEffect(Enum):
    """Smallest page refresh required by an operation transition."""

    NONE = "none"
    DIALOG = "dialog"
    CONTROLS = "controls"
    FULL = "full"


@dataclass(frozen=True, slots=True)
class WorkspaceOperationTransition:
    """Detached effect returned to the Qt composition page."""

    effect: WorkspaceRefreshEffect
    notice: str = ""
    reintegrate_reload: ReintegrateReloadDirective | None = None
    average_reload: AverageReloadDirective | None = None
    request_catalog: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.effect) is not WorkspaceRefreshEffect
            or type(self.notice) is not str
            or type(self.request_catalog) is not bool
            or (
                self.reintegrate_reload is not None
                and type(self.reintegrate_reload)
                is not ReintegrateReloadDirective
            )
            or (
                self.average_reload is not None
                and type(self.average_reload) is not AverageReloadDirective
            )
            or (
                self.reintegrate_reload is not None
                and self.average_reload is not None
            )
        ):
            raise ValueError("workspace operation transition is invalid")


@dataclass(frozen=True, slots=True)
class ReintegrateOperationState:
    """Active Reintegrate identity and its invalidated Browse owner."""

    identity: OperationIdentity
    capture: LoadedBrowseCapture
    dimension: str

    def __post_init__(self) -> None:
        if (
            type(self.identity) is not OperationIdentity
            or type(self.capture) is not LoadedBrowseCapture
            or self.dimension not in {"1d", "2d"}
        ):
            raise ValueError("Reintegrate operation state is invalid")


@dataclass(frozen=True, slots=True)
class AverageOperationState:
    """Active Average identity, intent boundary, and cleanup token."""

    identity: OperationIdentity
    revision: int
    target: str
    entry: str
    pending: OperationPending | None = None
    source_root: str | None = None

    def __post_init__(self) -> None:
        if (
            type(self.identity) is not OperationIdentity
            or type(self.revision) is not int
            or self.revision < 0
            or type(self.target) is not str
            or not self.target
            or type(self.entry) is not str
            or not self.entry
            or (
                self.source_root is not None
                and (
                    type(self.source_root) is not str
                    or not self.source_root
                    or not os.path.isabs(self.source_root)
                    or os.path.normcase(os.path.normpath(self.source_root))
                    != self.source_root
                )
            )
            or (
                self.pending is not None
                and (
                    type(self.pending) is not OperationPending
                    or self.pending.identity is not self.identity
                )
            )
        ):
            raise ValueError("Average operation state is invalid")


def _terminal_identity_for_target(
    value: object,
    target: str,
) -> StreamTerminal | None:
    """Return an exact writer seal only for its lexical output target."""

    if type(value) is not StreamTerminal or type(target) is not str or not target:
        return None
    if stream_terminal_object_revision(value) is None:
        return None
    normalized = os.path.normcase(os.path.abspath(os.path.expanduser(target)))
    return value if value.target == normalized else None


class WorkspaceOperationOwner:
    """Sole owner of the page's finite experiment-operation worker slot."""

    def __init__(self) -> None:
        self._slot = OperationSlot()
        self._reintegrate: ReintegrateOperationState | None = None
        self._average: AverageOperationState | None = None

    @property
    def owned(self) -> bool:
        return self._slot.owned

    @property
    def busy(self) -> bool:
        return self._slot.owned

    @property
    def current_identity(self) -> OperationIdentity | None:
        return self._slot.current_identity

    @property
    def reintegrate_identity(self) -> OperationIdentity | None:
        state = self._reintegrate
        return None if state is None else state.identity

    @property
    def reintegrate_state(self) -> ReintegrateOperationState | None:
        return self._reintegrate

    @property
    def reintegrate_dimension(self) -> str | None:
        state = self._reintegrate
        return None if state is None else state.dimension

    @property
    def reintegrate_capture(self) -> LoadedBrowseCapture | None:
        state = self._reintegrate
        return None if state is None else state.capture

    @property
    def average_identity(self) -> OperationIdentity | None:
        state = self._average
        return None if state is None else state.identity

    @property
    def average_state(self) -> AverageOperationState | None:
        return self._average

    @property
    def average_pending(self) -> OperationPending | None:
        state = self._average
        return None if state is None else state.pending

    def begin_calibrate(
        self, request: object, stamp: OperationContextStamp
    ) -> OperationIdentity | None:
        return self._slot.begin_calibrate(request, stamp)

    def begin_mask(
        self, request: object, stamp: OperationContextStamp
    ) -> OperationIdentity | None:
        return self._slot.begin_mask(request, stamp)

    def begin_asset_validation(
        self, request: object, stamp: OperationContextStamp
    ) -> OperationIdentity | None:
        return self._slot.begin_asset_validation(request, stamp)

    def begin_background(
        self,
        plan: object,
        stamp: OperationContextStamp,
        owner: object,
        reservation: object,
    ) -> OperationIdentity | None:
        return self._slot.begin_background(plan, stamp, owner, reservation)

    def begin_reintegrate(
        self,
        capture: LoadedBrowseCapture,
        *,
        dimension: str,
        preparation_values: Mapping[str, object],
        stamp: OperationContextStamp,
    ) -> OperationIdentity | None:
        if type(capture) is not LoadedBrowseCapture:
            return None
        identity = self._slot.begin_reintegrate(
            target=capture.target,
            entry=capture.entry,
            source_root=capture.request.source_root,
            expected_target_snapshot=capture.target_snapshot,
            expected_labels=capture.labels,
            dimension=dimension,
            preparation_values=preparation_values,
            stamp=stamp,
            expected_terminal_identity=(
                capture.request.terminal_commit_identity
                if stream_terminal_object_revision(
                    capture.request.terminal_commit_identity
                )
                is not None
                else None
            ),
        )
        if identity is None:
            return None
        self._reintegrate = ReintegrateOperationState(
            identity, capture, dimension
        )
        return identity

    def begin_average(
        self,
        configuration: FrozenRunConfiguration,
        target: str,
        *,
        revision: int,
        entry: str = "entry",
    ) -> OperationIdentity | None:
        if type(revision) is not int or revision < 0:
            return None
        source_root = configuration.project_root or None
        if source_root is not None and (
            type(source_root) is not str
            or not source_root
            or not os.path.isabs(source_root)
            or os.path.normcase(os.path.normpath(source_root)) != source_root
        ):
            return None
        identity = self._slot.begin_average(
            configuration,
            target,
            entry=entry,
            stamp=OperationContextStamp(revision),
        )
        if identity is None:
            return None
        self._average = AverageOperationState(
            identity,
            revision,
            str(target),
            entry,
            source_root=source_root,
        )
        return identity

    def cancel(self, identity: object) -> bool:
        return self._slot.cancel(identity)

    def cancel_average(self) -> bool:
        state = self._average
        return bool(
            state is not None and self._slot.cancel(state.identity)
        )

    def cancel_reintegrate(self, dimension: str) -> bool:
        state = self._reintegrate
        return bool(
            state is not None
            and state.dimension == dimension
            and self._slot.cancel(state.identity)
        )

    def retry_average(self) -> bool:
        state = self._average
        if state is None or state.pending is None:
            return False
        accepted = self._slot.retry_average(state.identity, state.pending)
        if accepted:
            self._average = replace(state, pending=None)
        return accepted

    def observe_stamp(
        self,
        stamp: OperationContextStamp,
        *,
        intent_revision: int,
    ) -> None:
        state = self._average
        if (
            state is not None
            and self._slot.current_identity is state.identity
        ):
            self._slot.observe_stamp(OperationContextStamp(intent_revision))
            return
        self._slot.observe_stamp(stamp)

    def poll(self, identity: object) -> OperationUpdate | None:
        return self._slot.poll(identity)

    def consume_reintegrate_update(
        self, update: object
    ) -> WorkspaceOperationTransition:
        state = self._reintegrate
        if (
            type(update) is not OperationUpdate
            or state is None
            or update.identity is not state.identity
        ):
            return WorkspaceOperationTransition(WorkspaceRefreshEffect.NONE)
        shown = {"1d": "1-D", "2d": "2-D"}.get(
            state.dimension, "operation"
        )
        if update.terminal is None:
            notice = ""
            if update.progress is not None:
                notice = (
                    f"Reintegrate {shown}: {update.progress.stage} "
                    f"{update.progress.completed}/{update.progress.total}…"
                )
            return WorkspaceOperationTransition(
                WorkspaceRefreshEffect.CONTROLS, notice
            )

        terminal = update.terminal
        capture = state.capture
        self._reintegrate = None
        result = terminal.payload
        notice = (
            f"Reintegrate {shown} {result.disposition.lower()}; "
            "reloading persisted results."
            if (
                terminal.status is OperationTerminalStatus.RETURNED
                and type(result) is ReintegrateResult
            )
            else f"Reintegrate {shown} cancelled; reloading persisted results."
            if terminal.status is OperationTerminalStatus.CANCELLED
            else f"Reintegrate {shown} failed: {terminal.diagnostic}"
        )
        committed = (
            terminal.status is OperationTerminalStatus.RETURNED
            and type(result) is ReintegrateResult
            and result.disposition == "COMMITTED"
        )
        commit_identity = (
            _terminal_identity_for_target(
                result.commit_identity, capture.target
            )
            if committed
            else None
        )
        if committed and commit_identity is None:
            notice = (
                f"Reintegrate {shown} returned an invalid commit identity; "
                "reloading the persisted artifact without its terminal seal."
            )
        directive = ReintegrateReloadDirective(
            capture.request, capture.target, commit_identity
        )
        return WorkspaceOperationTransition(
            WorkspaceRefreshEffect.FULL,
            notice,
            reintegrate_reload=directive,
        )

    def consume_average_update(
        self,
        update: object,
        *,
        current_intent_revision: int,
    ) -> WorkspaceOperationTransition:
        state = self._average
        if (
            type(update) is not OperationUpdate
            or state is None
            or update.identity is not state.identity
        ):
            return WorkspaceOperationTransition(WorkspaceRefreshEffect.NONE)
        if update.pending is not None:
            pending = update.pending
            if type(pending) is not OperationPending:
                return WorkspaceOperationTransition(
                    WorkspaceRefreshEffect.CONTROLS,
                    "Average failed: invalid cleanup-pending token",
                )
            self._average = replace(state, pending=pending)
            phase = pending.phase.replace("-", " ")
            return WorkspaceOperationTransition(
                WorkspaceRefreshEffect.CONTROLS,
                f"Average {phase} pending; press Run to retry or Stop to cancel.",
            )
        if update.terminal is None:
            notice = ""
            if update.progress is not None:
                notice = (
                    f"Average: {update.progress.stage} "
                    f"{update.progress.completed}/{update.progress.total}…"
                )
            return WorkspaceOperationTransition(
                WorkspaceRefreshEffect.CONTROLS, notice
            )

        target, entry, revision = state.target, state.entry, state.revision
        terminal = update.terminal
        result = terminal.payload
        from xrd_tools.reduction.average import AverageScanResult

        valid_result = False
        if type(result) is AverageScanResult:
            try:
                result.__post_init__()
            except (AttributeError, TypeError, ValueError, OverflowError):
                pass
            else:
                valid_result = True
        typed_cancellation = bool(
            valid_result and result.disposition == "CANCELLED"
        )
        if terminal.status is OperationTerminalStatus.CANCELLED:
            self._average = None
            notice = (
                "Average cancelled."
                if result is None or typed_cancellation
                else "Average failed: invalid terminal result"
            )
            return WorkspaceOperationTransition(
                WorkspaceRefreshEffect.CONTROLS, notice
            )
        if typed_cancellation:
            self._average = None
            return WorkspaceOperationTransition(
                WorkspaceRefreshEffect.CONTROLS,
                "Average failed: invalid terminal result",
            )
        if not valid_result:
            self._average = None
            return WorkspaceOperationTransition(
                WorkspaceRefreshEffect.CONTROLS,
                f"Average failed: {terminal.diagnostic or 'invalid terminal result'}",
            )
        stale = update.stale or revision != current_intent_revision
        if terminal.status is OperationTerminalStatus.FAILED:
            self._average = None
            notice = (
                f"{result.diagnostic_code}: {result.diagnostic}"
                if result.disposition == "ABORTED"
                else f"Average failed: {terminal.diagnostic}"
            )
            return WorkspaceOperationTransition(
                WorkspaceRefreshEffect.CONTROLS, notice
            )
        if result.disposition == "REFUSED":
            self._average = None
            return WorkspaceOperationTransition(
                WorkspaceRefreshEffect.CONTROLS,
                f"{result.diagnostic_code}: {result.diagnostic}",
            )
        if (
            result.disposition != "COMMITTED"
            or terminal.status is not OperationTerminalStatus.RETURNED
        ):
            self._average = None
            return WorkspaceOperationTransition(
                WorkspaceRefreshEffect.CONTROLS,
                "Average returned an invalid terminal disposition.",
            )
        if result.target != target or result.entry != entry:
            self._average = None
            return WorkspaceOperationTransition(
                WorkspaceRefreshEffect.CONTROLS,
                "Average terminal target mismatch; Browse was not reloaded.",
            )
        commit_identity = _terminal_identity_for_target(
            result.commit_identity, target
        )
        if commit_identity is None:
            self._average = None
            return WorkspaceOperationTransition(
                WorkspaceRefreshEffect.CONTROLS,
                "Average terminal commit identity mismatch; Browse was not reloaded.",
            )
        if stale:
            self._average = None
            return WorkspaceOperationTransition(
                WorkspaceRefreshEffect.CONTROLS,
                "Average committed but context changed; Browse was not reloaded.",
                request_catalog=True,
            )
        directive = AverageReloadDirective(
            target,
            entry,
            commit_identity,
            state.source_root,
        )
        self._average = None
        return WorkspaceOperationTransition(
            WorkspaceRefreshEffect.CONTROLS,
            "Average committed; Browse reload queued.",
            average_reload=directive,
            request_catalog=True,
        )

    def consume_lost_owner(
        self, identity: OperationIdentity
    ) -> WorkspaceOperationTransition:
        reintegrate = self._reintegrate
        if reintegrate is not None and reintegrate.identity is identity:
            shown = {"1d": "1-D", "2d": "2-D"}.get(
                reintegrate.dimension, "operation"
            )
            directive = ReintegrateReloadDirective(
                reintegrate.capture.request,
                reintegrate.capture.target,
            )
            self._reintegrate = None
            return WorkspaceOperationTransition(
                WorkspaceRefreshEffect.FULL,
                f"Reintegrate {shown} failed before terminal publication.",
                reintegrate_reload=directive,
            )
        average = self._average
        if average is not None and average.identity is identity:
            self._average = None
            return WorkspaceOperationTransition(
                WorkspaceRefreshEffect.CONTROLS,
                "Average failed before terminal publication.",
            )
        return WorkspaceOperationTransition(WorkspaceRefreshEffect.NONE)

    def close(self) -> OperationCleanupReceipt:
        receipt = self._slot.close()
        if receipt.cleanup_status is CleanupStatus.CLEANED:
            self._reintegrate = None
            self._average = None
        return receipt


__all__ = [
    "AverageOperationState",
    "ReintegrateOperationState",
    "WorkspaceOperationOwner",
    "WorkspaceOperationTransition",
    "WorkspaceRefreshEffect",
]
