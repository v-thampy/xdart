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
from xrd_tools.session.run_configuration import FrozenRunConfiguration

from .adapters.external_operation import OperationSlot
from .browse_values import LoadedBrowseCapture
from .events import CleanupStatus
from .operation_values import (
    OperationCleanupReceipt,
    OperationContextStamp,
    OperationIdentity,
    OperationPending,
    OperationProgress,
    OperationTerminalStatus,
    OperationUpdate,
)
from .processed_browser import (
    AverageReloadDirective,
    ReintegrateSuccessorDirective,
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
    reintegrate_successor: ReintegrateSuccessorDirective | None = None
    average_reload: AverageReloadDirective | None = None
    request_catalog: bool = False
    reintegrate_progress: OperationProgress | None = None

    def __post_init__(self) -> None:
        if (
            type(self.effect) is not WorkspaceRefreshEffect
            or type(self.notice) is not str
            or type(self.request_catalog) is not bool
            or (
                self.reintegrate_progress is not None
                and (
                    type(self.reintegrate_progress) is not OperationProgress
                    or self.effect is not WorkspaceRefreshEffect.NONE
                    or self.reintegrate_successor is not None
                    or self.average_reload is not None
                    or self.request_catalog
                )
            )
            or (
                self.reintegrate_successor is not None
                and type(self.reintegrate_successor)
                is not ReintegrateSuccessorDirective
            )
            or (
                self.average_reload is not None
                and type(self.average_reload) is not AverageReloadDirective
            )
            or (
                self.reintegrate_successor is not None
                and self.average_reload is not None
            )
        ):
            raise ValueError("workspace operation transition is invalid")


@dataclass(frozen=True, slots=True)
class ReintegrateOperationState:
    """Active immutable Reintegrate identity and its live predecessor."""

    identity: OperationIdentity
    capture: LoadedBrowseCapture
    dimension: str
    stamp: OperationContextStamp
    progress: OperationProgress | None = None
    cancel_accepted: bool = False
    owner_abandoned: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.identity) is not OperationIdentity
            or type(self.capture) is not LoadedBrowseCapture
            or self.dimension not in {"1d", "2d"}
            or type(self.stamp) is not OperationContextStamp
            or self.stamp.context_token
            != self.capture.selection.context_token
            or self.stamp.display_generation
            != self.capture.selection.display_generation
            or (
                self.progress is not None
                and (
                    type(self.progress) is not OperationProgress
                    or self.progress.identity is not self.identity
                )
            )
            or type(self.cancel_accepted) is not bool
            or type(self.owner_abandoned) is not bool
        ):
            raise ValueError("Reintegrate operation state is invalid")
        self.stamp.__post_init__()


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


def _reintegrate_result_notice_suffix(result: object) -> str:
    """Surface bounded fallback and private-cleanup diagnostics verbatim."""

    parts: list[str] = []
    diagnostics = getattr(result, "diagnostics", ())
    if type(diagnostics) is tuple:
        fallback = next(
            (
                value
                for value in diagnostics
                if type(value) is str
                and value.startswith("PREPARED_CAPSULE_MISS:")
            ),
            None,
        )
        if fallback is not None:
            parts.append(f" Prepared-data fallback: {fallback}.")
    hidden_orphan = getattr(result, "hidden_orphan", None)
    if type(hidden_orphan) is str and hidden_orphan:
        parts.append(
            " Manual cleanup is required for hidden candidate: "
            f"{hidden_orphan}."
        )
    return "".join(parts)


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
    def reintegrate_cancel_accepted(self) -> bool:
        state = self._reintegrate
        return bool(state is not None and state.cancel_accepted)

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
        if capture.prepared_reintegrate_offer is None:
            return None
        identity = self._slot.begin_reintegrate_successor(
            source_artifact=capture.target,
            entry=capture.entry,
            source_root=capture.request.source_root,
            expected_target_snapshot=capture.target_snapshot,
            expected_labels=capture.labels,
            dimension=dimension,
            preparation_values=preparation_values,
            prepared_offer=capture.prepared_reintegrate_offer,
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
            identity, capture, dimension, stamp,
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
        if (
            state is None
            or state.dimension != dimension
            or state.cancel_accepted
        ):
            return False
        accepted = self._slot.cancel(state.identity)
        if accepted and self._reintegrate is state:
            self._reintegrate = replace(state, cancel_accepted=True)
        return bool(accepted)

    def abandon_reintegrate(self, identity: object) -> bool:
        """Withdraw GUI adoption authority and best-effort stop exact work."""

        state = self._reintegrate
        if state is None or identity is not state.identity:
            return False
        accepted = (
            False
            if state.cancel_accepted
            else self._slot.cancel(state.identity)
        )
        if self._reintegrate is state:
            self._reintegrate = replace(
                state,
                cancel_accepted=state.cancel_accepted or bool(accepted),
                owner_abandoned=True,
            )
        return True

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
        reintegrate = self._reintegrate
        if (
            reintegrate is not None
            and self._slot.current_identity is reintegrate.identity
        ):
            self._slot.observe_stamp(stamp)
            if reintegrate.stamp != stamp:
                accepted = (
                    False
                    if reintegrate.cancel_accepted
                    else self._slot.cancel(reintegrate.identity)
                )
                if self._reintegrate is reintegrate:
                    self._reintegrate = replace(
                        reintegrate,
                        cancel_accepted=(
                            reintegrate.cancel_accepted or bool(accepted)
                        ),
                        owner_abandoned=True,
                    )
            return
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
            progress = update.progress
            prior = state.progress
            if (
                update.stale
                or state.cancel_accepted
                or progress is None
                or (
                    prior is not None
                    and (
                        progress.revision <= prior.revision
                        or progress.stage == prior.stage
                        and progress.completed < prior.completed
                    )
                )
            ):
                return WorkspaceOperationTransition(
                    WorkspaceRefreshEffect.NONE
                )
            self._reintegrate = replace(state, progress=progress)
            notice = (
                f"Reintegrate {shown}: {progress.stage} "
                f"{progress.completed}/{progress.total}…"
            )
            return WorkspaceOperationTransition(
                WorkspaceRefreshEffect.NONE,
                notice,
                reintegrate_progress=progress,
            )

        terminal = update.terminal
        capture = state.capture
        self._reintegrate = None
        result = terminal.payload
        if terminal.status is OperationTerminalStatus.CANCELLED:
            return WorkspaceOperationTransition(
                WorkspaceRefreshEffect.CONTROLS,
                f"Reintegrate {shown} stopped; the selected artifact is unchanged.",
            )
        if terminal.status is not OperationTerminalStatus.RETURNED:
            return WorkspaceOperationTransition(
                WorkspaceRefreshEffect.CONTROLS,
                f"Reintegrate {shown} failed: {terminal.diagnostic}",
            )
        # The worker imported and exact-gated the heavy successor result.  At
        # terminal this import is already resident and performs no file work.
        from xrd_tools.reduction import ReintegrateSuccessorResult
        if type(result) is not ReintegrateSuccessorResult:
            return WorkspaceOperationTransition(
                WorkspaceRefreshEffect.CONTROLS,
                f"Reintegrate {shown} returned an invalid result.",
            )
        if result.disposition == "ABORTED":
            return WorkspaceOperationTransition(
                WorkspaceRefreshEffect.CONTROLS,
                f"Reintegrate {shown} stopped; the selected artifact is unchanged."
                f"{_reintegrate_result_notice_suffix(result)}",
            )
        committed = result.disposition in {"COMMITTED", "ALREADY_COMMITTED"}
        terminal_identity = result.terminal
        valid = (
            committed
            and result.source_artifact == capture.target
            and result.output_artifact != capture.target
            and result.input_labels == capture.labels
            and type(result.committed_labels) is tuple
            and bool(result.committed_labels)
            and result.committed_labels
            == tuple(sorted(set(result.committed_labels)))
            and all(
                type(label) is int and label >= 0
                for label in result.committed_labels
            )
            and type(result.publication_dropped_labels) is tuple
            and result.publication_dropped_labels
            == tuple(sorted(set(result.publication_dropped_labels)))
            and all(
                type(label) is int and label >= 0
                for label in result.publication_dropped_labels
            )
            and not set(result.committed_labels).intersection(
                result.publication_dropped_labels
            )
            and tuple(sorted((
                *result.committed_labels,
                *result.publication_dropped_labels,
            ))) == capture.labels
            and type(result.audit_identity) is str
            and len(result.audit_identity) == 64
            and all(
                character in "0123456789abcdef"
                for character in result.audit_identity
            )
            and result.hidden_orphan is None
            and type(terminal_identity) is StreamTerminal
            and stream_terminal_object_revision(terminal_identity) is not None
            and terminal_identity.target == result.output_artifact
            and type(result.commit_identity) is str
            and len(result.commit_identity) == 64
            and state.stamp is not None
        )
        if not valid:
            return WorkspaceOperationTransition(
                WorkspaceRefreshEffect.CONTROLS,
                f"Reintegrate {shown} returned invalid successor authority; "
                "the selected artifact is unchanged.",
            )
        if update.stale or state.owner_abandoned:
            return WorkspaceOperationTransition(
                WorkspaceRefreshEffect.CONTROLS,
                f"Reintegrate {shown} published a new version after its "
                "display owner changed; it was cataloged without switching."
                f"{_reintegrate_result_notice_suffix(result)}",
                request_catalog=True,
            )
        directive = ReintegrateSuccessorDirective(
            capture,
            result.output_artifact,
            capture.entry,
            result.committed_labels,
            terminal_identity,
            result.version_identity,
            result.publication_identity,
            result.operation_identity,
            result.commit_identity,
            state.identity,
            state.stamp,
        )
        # UNREACHABLE SINCE 2026-09-04, and kept honest anyway.  Nothing
        # produces ALREADY_COMMITTED now that an occupied slot is replaced
        # rather than reused, but a lying string in a dead branch is still a
        # lying string the next reader has to disbelieve.  The enum member and
        # its consumers are retired with the rest of the dead reuse machinery
        # (item 7), not mid-packet.
        action = (
            "replaced the existing version"
            if result.disposition == "ALREADY_COMMITTED"
            else "published a new version"
        )
        return WorkspaceOperationTransition(
            WorkspaceRefreshEffect.CONTROLS,
            f"Reintegrate {shown} {action}; validating it before switching."
            f"{_reintegrate_result_notice_suffix(result)}",
            reintegrate_successor=directive,
            request_catalog=True,
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
            # PUBLISHED_UNVERIFIED is a FAILURE THAT CHANGED THE FOLDER: the
            # atomic replacement landed and only the check after it did not, so
            # a new file really is at the slot.  Every other failure here leaves
            # the directory as it was, and CONTROLS is right for those.  Fable
            # F4 on `55338dd9`: refreshing only the controls left the operator
            # told about a file the browser would not show them until something
            # else happened to refresh it.
            return WorkspaceOperationTransition(
                WorkspaceRefreshEffect.FULL
                if result.disposition == "PUBLISHED_UNVERIFIED"
                else WorkspaceRefreshEffect.CONTROLS,
                notice,
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
        # Average never writes `target`: it is only this operation's directory
        # and naming anchor.  Recompute the stable slot from that anchor so a
        # terminal naming a different directory, family or operation cannot be
        # adopted.  Path math only -- the worker already did the full
        # provenance verification.
        #
        # NARROWED by the stable-slot policy, deliberately.  The recompute used
        # to include the result's own version identity, so it also refused a
        # wrong VERSION.  Public names no longer carry one, so this guard can
        # no longer see that class of mismatch.  Nothing was lost silently, but
        # name the carrier precisely: the recomputed SLOT no longer encodes a
        # version, so what actually verifies version agreement is the persisted
        # `operation_identity` comparison in `_committed_average_mismatch`,
        # because `operation_identity` is a hash over a payload that embeds both
        # `version_identity` and `output_artifact` (average.py:475/564/596).
        # This guard's remaining job is to refuse a FOREIGN artifact, which is
        # the failure this branch was written for.
        from xrd_tools.reduction.average import (
            _average_output_artifact,
            _average_target,
        )

        try:
            successor = _average_output_artifact(_average_target(target))
        except Exception:
            successor = None
        if successor is None or result.target != successor or result.entry != entry:
            self._average = None
            return WorkspaceOperationTransition(
                WorkspaceRefreshEffect.CONTROLS,
                "Average terminal target mismatch; Browse was not reloaded.",
            )
        commit_identity = _terminal_identity_for_target(
            result.commit_identity, successor
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
            successor,
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
            self._reintegrate = None
            return WorkspaceOperationTransition(
                WorkspaceRefreshEffect.CONTROLS,
                f"Reintegrate {shown} failed before terminal publication; "
                "the selected artifact is unchanged.",
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
