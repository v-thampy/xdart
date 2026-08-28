"""Qt-free ownership for workspace experiment operations.

The page composes notices, dialogs, and Browse mutations.  This owner keeps
the one experiment worker slot and the state that must survive those page
effects: Average cleanup commands and Reintegrate's mandatory reload intent.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import os
from typing import Callable, Mapping

from xdart.modules.display_context import BrowseContext, ContextKind, DisplaySelection
from xrd_tools.io.output_transaction import (
    StreamTerminal,
    TargetSnapshot,
    stream_terminal_object_revision,
)
from xrd_tools.reduction import ReintegrateResult
from xrd_tools.session.run_configuration import FrozenRunConfiguration

from .adapters.external_operation import OperationSlot
from .browse_values import BrowseLoadRequest
from .events import CleanupStatus
from .operation_values import (
    OperationCleanupReceipt,
    OperationContextStamp,
    OperationIdentity,
    OperationPending,
    OperationTerminalStatus,
    OperationUpdate,
)


class WorkspaceRefreshEffect(Enum):
    """Smallest page refresh required by an operation transition."""

    NONE = "none"
    DIALOG = "dialog"
    CONTROLS = "controls"
    FULL = "full"

    def __bool__(self) -> bool:
        return self is not WorkspaceRefreshEffect.NONE


@dataclass(frozen=True, slots=True)
class ReintegrateBrowseCapture:
    """One exact, stable processed-Browse source for Reintegrate."""

    context: BrowseContext
    request: BrowseLoadRequest
    selection: DisplaySelection
    target: str
    entry: str
    target_snapshot: TargetSnapshot
    labels: tuple[int, ...]

    def __post_init__(self) -> None:
        valid = (
            type(self.context) is BrowseContext
            and type(self.request) is BrowseLoadRequest
            and type(self.selection) is DisplaySelection
            and self.selection.kind is ContextKind.BROWSE
            and type(self.target) is str
            and bool(self.target)
            and self.request.source_path == self.target
            and type(self.entry) is str
            and bool(self.entry)
            and type(self.target_snapshot) is TargetSnapshot
            and self.target_snapshot.exists
            and type(self.labels) is tuple
            and bool(self.labels)
            and self.labels == tuple(sorted(set(self.labels)))
            and all(type(label) is int and label >= 0 for label in self.labels)
        )
        if not valid:
            raise ValueError("Reintegrate Browse capture is invalid")

    def is_exactly(self, other: object) -> bool:
        """Compare value facts but require all live owners by identity."""

        return bool(
            type(other) is ReintegrateBrowseCapture
            and other.context is self.context
            and other.request is self.request
            and other.selection is self.selection
            and other.target == self.target
            and other.entry == self.entry
            and other.target_snapshot == self.target_snapshot
            and other.labels == self.labels
        )


@dataclass(frozen=True, slots=True)
class ReintegrateReloadDirective:
    """Persisted Browse artifact that must be reloaded after invalidation."""

    request: BrowseLoadRequest
    target: str
    terminal_commit_identity: StreamTerminal | None = None

    def __post_init__(self) -> None:
        if (
            type(self.request) is not BrowseLoadRequest
            or type(self.target) is not str
            or not self.target
            or self.request.source_path != self.target
            or (
                self.terminal_commit_identity is not None
                and stream_terminal_object_revision(
                    self.terminal_commit_identity
                )
                is None
            )
        ):
            raise ValueError("Reintegrate reload directive is invalid")


@dataclass(frozen=True, slots=True)
class AverageReloadDirective:
    """Validated committed Average result that the page may Browse."""

    target: str
    entry: str
    terminal_commit_identity: StreamTerminal

    def __post_init__(self) -> None:
        if (
            type(self.target) is not str
            or not self.target
            or type(self.entry) is not str
            or not self.entry
            or stream_terminal_object_revision(self.terminal_commit_identity)
            is None
        ):
            raise ValueError("Average reload directive is invalid")


@dataclass(frozen=True, slots=True)
class WorkspaceOperationTransition:
    """Detached effect returned to the Qt composition page."""

    effect: WorkspaceRefreshEffect
    notice: str = ""
    average_reload: AverageReloadDirective | None = None
    request_catalog: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.effect) is not WorkspaceRefreshEffect
            or type(self.notice) is not str
            or type(self.request_catalog) is not bool
            or (
                self.average_reload is not None
                and type(self.average_reload) is not AverageReloadDirective
            )
        ):
            raise ValueError("workspace operation transition is invalid")


@dataclass(frozen=True, slots=True)
class ReintegrateOperationState:
    """Active Reintegrate identity and its invalidated Browse owner."""

    identity: OperationIdentity
    capture: ReintegrateBrowseCapture
    dimension: str

    def __post_init__(self) -> None:
        if (
            type(self.identity) is not OperationIdentity
            or type(self.capture) is not ReintegrateBrowseCapture
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
        self._reintegrate_reload: ReintegrateReloadDirective | None = None
        self._average: AverageOperationState | None = None

    @property
    def owned(self) -> bool:
        return self._slot.owned

    @property
    def busy(self) -> bool:
        return self._slot.owned or self._reintegrate_reload is not None

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
    def reintegrate_capture(self) -> ReintegrateBrowseCapture | None:
        state = self._reintegrate
        return None if state is None else state.capture

    @property
    def pending_reintegrate_reload(
        self,
    ) -> ReintegrateReloadDirective | None:
        return self._reintegrate_reload

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

    def begin(
        self,
        frozen: object,
        stamp: OperationContextStamp,
        body: Callable[..., object],
    ) -> OperationIdentity | None:
        return self._slot._begin(frozen, stamp, body)

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
        capture: ReintegrateBrowseCapture,
        *,
        dimension: str,
        preparation_values: Mapping[str, object],
        stamp: OperationContextStamp,
    ) -> OperationIdentity | None:
        if type(capture) is not ReintegrateBrowseCapture:
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
            self.require_reintegrate_reload(capture)
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
        identity = self._slot.begin_average(
            configuration,
            target,
            entry=entry,
            stamp=OperationContextStamp(revision),
        )
        if identity is None:
            return None
        self._average = AverageOperationState(
            identity, revision, str(target), entry
        )
        return identity

    def require_reintegrate_reload(
        self,
        capture: ReintegrateBrowseCapture,
        terminal_commit_identity: StreamTerminal | None = None,
    ) -> ReintegrateReloadDirective:
        directive = ReintegrateReloadDirective(
            capture.request,
            capture.target,
            terminal_commit_identity,
        )
        self._reintegrate_reload = directive
        return directive

    def retire_reintegrate_reload(
        self, directive: ReintegrateReloadDirective
    ) -> bool:
        if self._reintegrate_reload is not directive:
            return False
        self._reintegrate_reload = None
        return True

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
        self.require_reintegrate_reload(capture, commit_identity)
        return WorkspaceOperationTransition(
            WorkspaceRefreshEffect.FULL, notice
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
                    WorkspaceRefreshEffect.FULL,
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
        self._average = None
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
            notice = (
                "Average cancelled."
                if result is None or typed_cancellation
                else "Average failed: invalid terminal result"
            )
            return WorkspaceOperationTransition(
                WorkspaceRefreshEffect.FULL, notice
            )
        if typed_cancellation:
            return WorkspaceOperationTransition(
                WorkspaceRefreshEffect.FULL,
                "Average failed: invalid terminal result",
            )
        if not valid_result:
            return WorkspaceOperationTransition(
                WorkspaceRefreshEffect.FULL,
                f"Average failed: {terminal.diagnostic or 'invalid terminal result'}",
            )
        stale = update.stale or revision != current_intent_revision
        if terminal.status is OperationTerminalStatus.FAILED:
            notice = (
                f"{result.diagnostic_code}: {result.diagnostic}"
                if result.disposition == "ABORTED"
                else f"Average failed: {terminal.diagnostic}"
            )
            return WorkspaceOperationTransition(
                WorkspaceRefreshEffect.FULL, notice
            )
        if result.disposition == "REFUSED":
            return WorkspaceOperationTransition(
                WorkspaceRefreshEffect.FULL,
                f"{result.diagnostic_code}: {result.diagnostic}",
            )
        if (
            result.disposition != "COMMITTED"
            or terminal.status is not OperationTerminalStatus.RETURNED
        ):
            return WorkspaceOperationTransition(
                WorkspaceRefreshEffect.FULL,
                "Average returned an invalid terminal disposition.",
            )
        if result.target != target or result.entry != entry:
            return WorkspaceOperationTransition(
                WorkspaceRefreshEffect.FULL,
                "Average terminal target mismatch; Browse was not reloaded.",
            )
        commit_identity = _terminal_identity_for_target(
            result.commit_identity, target
        )
        if commit_identity is None:
            return WorkspaceOperationTransition(
                WorkspaceRefreshEffect.FULL,
                "Average terminal commit identity mismatch; Browse was not reloaded.",
            )
        if stale:
            return WorkspaceOperationTransition(
                WorkspaceRefreshEffect.FULL,
                "Average committed but context changed; Browse was not reloaded.",
                request_catalog=True,
            )
        return WorkspaceOperationTransition(
            WorkspaceRefreshEffect.FULL,
            average_reload=AverageReloadDirective(
                target, entry, commit_identity
            ),
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
            self.require_reintegrate_reload(reintegrate.capture)
            self._reintegrate = None
            return WorkspaceOperationTransition(
                WorkspaceRefreshEffect.FULL,
                f"Reintegrate {shown} failed before terminal publication.",
            )
        average = self._average
        if average is not None and average.identity is identity:
            self._average = None
            return WorkspaceOperationTransition(
                WorkspaceRefreshEffect.FULL,
                "Average failed before terminal publication.",
            )
        return WorkspaceOperationTransition(WorkspaceRefreshEffect.NONE)

    def close(self) -> OperationCleanupReceipt:
        self._reintegrate_reload = None
        receipt = self._slot.close()
        if receipt.cleanup_status is CleanupStatus.CLEANED:
            self._reintegrate = None
            self._average = None
        return receipt


__all__ = [
    "AverageOperationState",
    "AverageReloadDirective",
    "ReintegrateBrowseCapture",
    "ReintegrateOperationState",
    "ReintegrateReloadDirective",
    "WorkspaceOperationOwner",
    "WorkspaceOperationTransition",
    "WorkspaceRefreshEffect",
]
