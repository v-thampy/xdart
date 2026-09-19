"""Qt-free exact owner for authored calibration and detector-mask assets.

The whole operation -> confirmation -> validation -> intent-CAS -> close
lifecycle lives under one outer tagged state. Qt objects stay page-owned and
are addressed only by monotonic typed identities in detached commands.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import os
from pathlib import Path
import stat

from xrd_tools.session.intent_store import (
    IntentCommitAccepted,
    IntentRecaptureRequired,
    RunIntentSnapshot,
    RunIntentStore,
)

from .contracts import SourceFileState
from .controls_editing import EditNoChange, EditRefusal, reduce_control_edit
from .controls_inventory import MASK_FILE, PONI_FILE
from .events import CleanupStatus
from .experiment_authoring import (
    AssetValidationRequest,
    AssetValidationResult,
    AuthoredAssetCandidate,
    CalibrationRequest,
    CalibrationResult,
    MaskRequest,
    authoring_source_context_current,
    mask_terminal_result_valid,
)
from .operation_values import (
    OperationCleanupReceipt,
    OperationContextStamp,
    OperationIdentity,
    OperationTerminalStatus,
    OperationUpdate,
)


class AuthoredAssetOwnerLifecycle(Enum):
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"


class AuthoredAssetPhase(Enum):
    IDLE = "idle"
    RUNNING = "running"
    TERMINAL_READY = "terminal_ready"
    CONFIRM_ISSUED = "confirm_issued"
    CONFIRM_PRESENTED = "confirm_presented"
    VALIDATING = "validating"
    DISMISSING = "dismissing"


class ClosingCleanupStatus(Enum):
    AWAITING = "awaiting"
    PENDING = "pending"
    CLEANED = "cleaned"


class AuthoredAssetRefreshEffect(Enum):
    NONE = "none"
    DIALOG = "dialog"
    CONTROLS = "controls"


class AuthoredAssetDialogEffect(Enum):
    OPEN = "open"
    CLOSE = "close"
    SET_BUSY = "set_busy"
    SET_IDLE = "set_idle"


@dataclass(frozen=True, slots=True, order=True)
class AuthoredAssetEvidenceIdentity:
    serial: int

    def __post_init__(self) -> None:
        if type(self.serial) is not int or self.serial < 1:
            raise ValueError("authored-asset evidence identity is invalid")


@dataclass(frozen=True, slots=True, order=True)
class AuthoredAssetDialogIdentity:
    serial: int

    def __post_init__(self) -> None:
        if type(self.serial) is not int or self.serial < 1:
            raise ValueError("authored-asset dialog identity is invalid")


@dataclass(frozen=True, slots=True)
class AuthoredAssetDialogIssue:
    identity: AuthoredAssetDialogIdentity
    asset: str
    paths: tuple[str, ...]
    source_directory: str

    def __post_init__(self) -> None:
        if (
            type(self.identity) is not AuthoredAssetDialogIdentity
            or self.asset not in {"poni", "mask"}
            or type(self.paths) is not tuple
            or any(
                type(path) is not str or not os.path.isabs(path)
                for path in self.paths
            )
            or type(self.source_directory) is not str
            or not os.path.isabs(self.source_directory)
        ):
            raise ValueError("authored-asset dialog issue is invalid")


@dataclass(frozen=True, slots=True)
class AuthoredAssetDialogCommand:
    identity: AuthoredAssetDialogIdentity
    effect: AuthoredAssetDialogEffect

    def __post_init__(self) -> None:
        if (
            type(self.identity) is not AuthoredAssetDialogIdentity
            or type(self.effect) is not AuthoredAssetDialogEffect
        ):
            raise ValueError("authored-asset dialog command is invalid")


@dataclass(frozen=True, slots=True)
class AuthoredAssetAdoption:
    before: RunIntentSnapshot
    result: IntentCommitAccepted | IntentRecaptureRequired
    path: str
    asset: str
    remember_path: bool
    dialog_identity: AuthoredAssetDialogIdentity | None

    def __post_init__(self) -> None:
        if (
            type(self.before) is not RunIntentSnapshot
            or type(self.result)
            not in {IntentCommitAccepted, IntentRecaptureRequired}
            or type(self.path) is not str
            or not os.path.isabs(self.path)
            or self.asset not in {"poni", "mask"}
            or type(self.remember_path) is not bool
            or self.remember_path
            != (type(self.result) is IntentCommitAccepted)
            or not (type(self.dialog_identity) is AuthoredAssetDialogIdentity
                    or self.asset == "mask" and self.dialog_identity is None)
        ):
            raise ValueError("authored-asset adoption is invalid")


@dataclass(frozen=True, slots=True)
class AuthoredAssetTransition:
    refresh: AuthoredAssetRefreshEffect = AuthoredAssetRefreshEffect.NONE
    notice: str = ""
    issue: AuthoredAssetDialogIssue | None = None
    dialog: AuthoredAssetDialogCommand | None = None
    cancel_identity: OperationIdentity | None = None
    adoption: AuthoredAssetAdoption | None = None
    error: bool = False
    mask_set_path: str | None = None

    def __post_init__(self) -> None:
        valid = (
            type(self.refresh) is AuthoredAssetRefreshEffect
            and type(self.notice) is str
            and type(self.error) is bool
            and (self.mask_set_path is None or (
                type(self.mask_set_path) is str
                and os.path.isabs(self.mask_set_path)
            ))
            and (
                self.issue is None
                or type(self.issue) is AuthoredAssetDialogIssue
            )
            and (
                self.dialog is None
                or type(self.dialog) is AuthoredAssetDialogCommand
            )
            and (
                self.cancel_identity is None
                or type(self.cancel_identity) is OperationIdentity
            )
            and (
                self.adoption is None
                or type(self.adoption) is AuthoredAssetAdoption
            )
            and not (self.issue is not None and self.dialog is not None)
        )
        if self.issue is not None:
            valid = valid and (
                self.refresh is AuthoredAssetRefreshEffect.DIALOG
                and self.cancel_identity is None
                and self.adoption is None
            )
        if self.dialog is not None:
            valid = valid and self.refresh is AuthoredAssetRefreshEffect.DIALOG
        if self.cancel_identity is not None and self.dialog is not None:
            valid = valid and (
                self.dialog.effect is AuthoredAssetDialogEffect.CLOSE
                and self.adoption is None
            )
        if self.adoption is not None:
            valid = valid and (
                self.cancel_identity is None
                and self.issue is None
                and (
                    self.refresh is AuthoredAssetRefreshEffect.DIALOG
                    and self.dialog is not None
                    and self.dialog.effect is AuthoredAssetDialogEffect.CLOSE
                    and self.dialog.identity is self.adoption.dialog_identity
                    or self.adoption.asset == "mask"
                    and self.adoption.dialog_identity is None
                    and self.dialog is None
                    and self.refresh is AuthoredAssetRefreshEffect.CONTROLS
                )
            )
        if self.refresh is AuthoredAssetRefreshEffect.NONE:
            valid = valid and (
                self.notice == ""
                and self.mask_set_path is None
                and self.issue is None
                and self.dialog is None
                and self.cancel_identity is None
                and self.adoption is None
            )
        if not valid:
            raise ValueError("authored-asset transition is invalid")


@dataclass(frozen=True, slots=True)
class _OpenState:
    phase: AuthoredAssetPhase = AuthoredAssetPhase.IDLE
    asset: str | None = None
    stamp: OperationContextStamp | None = None
    source_directory: str | None = None
    source_request: CalibrationRequest | MaskRequest | None = None
    expected_shape: tuple[int, int] | None = None
    operation_identity: OperationIdentity | None = None
    candidates: tuple[AuthoredAssetCandidate, ...] = ()
    evidence_identity: AuthoredAssetEvidenceIdentity | None = None
    dialog_identity: AuthoredAssetDialogIdentity | None = None
    validation_request: AssetValidationRequest | None = None


@dataclass(frozen=True, slots=True)
class _ClosingState:
    cleanup_identity: OperationIdentity | None
    cleanup_status: ClosingCleanupStatus
    dialog_identity: AuthoredAssetDialogIdentity | None
    terminal_seen: bool = False
    lost_owner_seen: bool = False


@dataclass(frozen=True, slots=True)
class _ClosedState:
    cleanup_identity: OperationIdentity | None
    terminal_seen: bool = False
    lost_owner_seen: bool = False


_OwnerState = _OpenState | _ClosingState | _ClosedState


def _valid_detached_value(value: object, expected_type: type) -> bool:
    """Revalidate an exact detached envelope before observing its fields."""

    if type(value) is not expected_type:
        return False
    try:
        value.__post_init__()
    except (AttributeError, TypeError, ValueError):
        return False
    return True


def _context_current(state: _OpenState, current_stamp: object) -> bool:
    try:
        return (
            type(current_stamp) is OperationContextStamp
            and current_stamp == state.stamp
            and state.source_request is not None
            and authoring_source_context_current(state.source_request)
        )
    except (AttributeError, OSError, TypeError, ValueError):
        return False


def _candidate_current(candidate: object) -> bool:
    try:
        if type(candidate) is not AuthoredAssetCandidate:
            return False
        candidate.__post_init__()
        path = Path(candidate.path)
        raw = path.lstat()
        expected = candidate.state
        observed = (
            raw.st_dev,
            raw.st_ino,
            raw.st_size,
            raw.st_mtime_ns,
            raw.st_ctime_ns,
        )
        return (
            not stat.S_ISLNK(raw.st_mode)
            and stat.S_ISREG(raw.st_mode)
            and observed
            == (
                expected.device,
                expected.inode,
                expected.size,
                expected.mtime_ns,
                expected.ctime_ns,
            )
            and SourceFileState.capture(path) == expected
        )
    except (OSError, TypeError, ValueError):
        return False


def _valid_operation_request(
    asset: object,
    request: object,
    stamp: object,
    identity: object,
) -> bool:
    if (
        asset not in {"poni", "mask"}
        or type(stamp) is not OperationContextStamp
        or type(identity) is not OperationIdentity
        or type(request) not in {CalibrationRequest, MaskRequest}
        or (asset == "poni") != (type(request) is CalibrationRequest)
    ):
        return False
    try:
        stamp.__post_init__()
        request.__post_init__()
    except (AttributeError, TypeError, ValueError):
        return False
    return True


def _terminal_evidence(
    state: _OpenState,
    update: OperationUpdate,
) -> tuple[tuple[AuthoredAssetCandidate, ...], tuple[int, int] | None] | None:
    terminal = update.terminal
    request = state.source_request
    if (
        terminal is None
        or terminal.status is not OperationTerminalStatus.RETURNED
    ):
        return None
    result = terminal.payload
    if state.asset == "poni":
        try:
            expected_argv = (
                (request.executable, request.source_path)
                if (
                    type(request) is CalibrationRequest
                    and Path(request.source_path).suffix.casefold()
                    not in {".h5", ".hdf5", ".nxs", ".nexus"}
                )
                else (request.executable, request.exact_hdf_url)
                if (
                    type(request) is CalibrationRequest
                    and request.exact_hdf_url is not None
                )
                else (request.executable,)
                if type(request) is CalibrationRequest
                else ()
            )
            valid = (
                type(result) is CalibrationResult
                and result.request is request
                and result.exit_code == 0
                and result.argv == expected_argv
                and result.diagnostic == ""
            )
            if valid:
                terminal.__post_init__()
                result.__post_init__()
            if not valid:
                return None
            candidates = tuple(
                AuthoredAssetCandidate(
                    "poni",
                    candidate.path,
                    candidate.proof,
                    candidate.proof.state,
                )
                for candidate in result.candidates
            )
            return candidates, None
        except (AttributeError, TypeError, ValueError, OverflowError):
            return None
    if state.asset == "mask" and type(request) is MaskRequest:
        try:
            if not mask_terminal_result_valid(terminal, request):
                return None
            candidate = AuthoredAssetCandidate(
                "mask",
                result.request.final_path,
                result.proof,
                result.final_state,
                result.request.source_path,
            )
            return (candidate,), result.proof.shape
        except (AttributeError, TypeError, ValueError, OverflowError):
            return None
    return None


class AuthoredAssetOwner:
    """Single exact owner with absorbing closing and closed states."""

    def __init__(self, intents: RunIntentStore) -> None:
        if not isinstance(intents, RunIntentStore):
            raise TypeError("authored-asset intent store is invalid")
        self._intents = intents
        self._state: _OwnerState = _OpenState()
        self._next_evidence_serial = 1
        self._next_dialog_serial = 1

    @property
    def lifecycle(self) -> AuthoredAssetOwnerLifecycle:
        state = self._state
        if type(state) is _OpenState:
            return AuthoredAssetOwnerLifecycle.OPEN
        if type(state) is _ClosingState:
            return AuthoredAssetOwnerLifecycle.CLOSING
        return AuthoredAssetOwnerLifecycle.CLOSED

    @property
    def phase(self) -> AuthoredAssetPhase | None:
        state = self._state
        return state.phase if type(state) is _OpenState else None

    @property
    def busy(self) -> bool:
        state = self._state
        return (
            type(state) is _OpenState
            and state.phase is not AuthoredAssetPhase.IDLE
        )

    @property
    def asset(self) -> str | None:
        state = self._state
        return state.asset if type(state) is _OpenState else None

    @property
    def source_directory(self) -> str | None:
        state = self._state
        return state.source_directory if type(state) is _OpenState else None

    @property
    def operation_identity(self) -> OperationIdentity | None:
        state = self._state
        return (
            state.operation_identity
            if type(state) is _OpenState
            else state.cleanup_identity
            if type(state) in {_ClosingState, _ClosedState}
            else None
        )

    @property
    def cleanup_identity(self) -> OperationIdentity | None:
        state = self._state
        return (
            state.cleanup_identity
            if type(state) in {_ClosingState, _ClosedState}
            else None
        )

    @property
    def evidence_identity(self) -> AuthoredAssetEvidenceIdentity | None:
        state = self._state
        return state.evidence_identity if type(state) is _OpenState else None

    @property
    def dialog_identity(self) -> AuthoredAssetDialogIdentity | None:
        state = self._state
        return (
            state.dialog_identity
            if type(state) in {_OpenState, _ClosingState}
            else None
        )

    @property
    def closing_cleanup_status(self) -> ClosingCleanupStatus | None:
        state = self._state
        if type(state) is _ClosingState:
            return state.cleanup_status
        if type(state) is _ClosedState:
            return ClosingCleanupStatus.CLEANED
        return None

    @property
    def closing_terminal_seen(self) -> bool:
        state = self._state
        return bool(
            type(state) in {_ClosingState, _ClosedState}
            and state.terminal_seen
        )

    def adopt_operation(
        self,
        asset: object,
        request: object,
        stamp: object,
        identity: object,
    ) -> AuthoredAssetTransition:
        state = self._state
        if (
            type(state) is not _OpenState
            or state.phase is not AuthoredAssetPhase.IDLE
            or not _valid_operation_request(asset, request, stamp, identity)
        ):
            return AuthoredAssetTransition()
        directory = (
            request.monitored_directory
            if type(request) is CalibrationRequest
            else request.source_directory
        )
        self._state = _OpenState(
            phase=AuthoredAssetPhase.RUNNING,
            asset=asset,
            stamp=stamp,
            source_directory=directory,
            source_request=request,
            operation_identity=identity,
        )
        return AuthoredAssetTransition(
            AuthoredAssetRefreshEffect.CONTROLS,
            f"{'Calibration' if asset == 'poni' else 'Mask'} operation started.",
        )

    def consume_operation_update(
        self,
        update: object,
        current_stamp: object,
    ) -> AuthoredAssetTransition:
        if not _valid_detached_value(update, OperationUpdate):
            return AuthoredAssetTransition()
        state = self._state
        if type(state) is _ClosingState:
            if (
                type(state.cleanup_identity) is OperationIdentity
                and update.identity is state.cleanup_identity
                and update.terminal is not None
            ):
                self._state = replace(state, terminal_seen=True)
            return AuthoredAssetTransition()
        if (
            type(state) is not _OpenState
            or update.identity is not state.operation_identity
        ):
            return AuthoredAssetTransition()
        if state.phase is AuthoredAssetPhase.RUNNING:
            return self._consume_process_update(state, update, current_stamp)
        if state.phase is AuthoredAssetPhase.VALIDATING:
            return self._consume_validation_update(state, update, current_stamp)
        if state.phase is AuthoredAssetPhase.DISMISSING:
            if update.terminal is None:
                return AuthoredAssetTransition()
            self._state = _OpenState()
            return AuthoredAssetTransition(
                AuthoredAssetRefreshEffect.CONTROLS,
                "Authored asset was not adopted.",
            )
        return AuthoredAssetTransition()

    def _consume_process_update(
        self,
        state: _OpenState,
        update: OperationUpdate,
        current_stamp: object,
    ) -> AuthoredAssetTransition:
        terminal = update.terminal
        label = "Calibration" if state.asset == "poni" else "Mask"
        if terminal is None:
            if update.progress is None:
                return AuthoredAssetTransition()
            return AuthoredAssetTransition(
                AuthoredAssetRefreshEffect.CONTROLS,
                f"{label}: {update.progress.stage}…",
            )
        if terminal.status is not OperationTerminalStatus.RETURNED:
            self._state = _OpenState()
            notice = (
                f"{label} cancelled."
                if terminal.status is OperationTerminalStatus.CANCELLED
                else f"{label} failed: {terminal.diagnostic}"
            )
            return AuthoredAssetTransition(
                AuthoredAssetRefreshEffect.CONTROLS, notice,
                error=terminal.status is OperationTerminalStatus.FAILED,
            )
        evidence = _terminal_evidence(state, update)
        if evidence is None:
            self._state = _OpenState()
            return AuthoredAssetTransition(
                AuthoredAssetRefreshEffect.CONTROLS,
                f"{label} returned without exact publication proof.",
            )
        if update.stale or not _context_current(state, current_stamp):
            self._state = _OpenState()
            return AuthoredAssetTransition(
                AuthoredAssetRefreshEffect.CONTROLS,
                "Authored asset finished but context changed; nothing was adopted.",
            )
        evidence_identity = AuthoredAssetEvidenceIdentity(
            self._next_evidence_serial
        )
        self._next_evidence_serial += 1
        candidates, expected_shape = evidence
        self._state = replace(
            state,
            phase=AuthoredAssetPhase.TERMINAL_READY,
            operation_identity=None,
            candidates=candidates,
            expected_shape=expected_shape,
            evidence_identity=evidence_identity,
        )
        notice = (
            "Choose whether to adopt the authored PONI."
            if state.asset == "poni" and candidates
            else "No new valid PONI was found; choose an existing PONI or cancel."
            if state.asset == "poni"
            else "Mask saved; updating Mask File…"
        )
        return AuthoredAssetTransition(
            AuthoredAssetRefreshEffect.CONTROLS, notice
        )

    def issue_confirmation(
        self, evidence: object, current_stamp: object
    ) -> AuthoredAssetTransition:
        state = self._state
        if (
            type(state) is not _OpenState
            or state.phase is not AuthoredAssetPhase.TERMINAL_READY
            or evidence is not state.evidence_identity
            or state.asset != "poni"
            or state.source_directory is None
        ):
            return AuthoredAssetTransition()
        if (
            not _context_current(state, current_stamp)
            or not all(_candidate_current(item) for item in state.candidates)
        ):
            self._state = _OpenState()
            return AuthoredAssetTransition(
                AuthoredAssetRefreshEffect.CONTROLS,
                "Authored asset context changed before confirmation.",
            )
        identity = AuthoredAssetDialogIdentity(self._next_dialog_serial)
        self._next_dialog_serial += 1
        issue = AuthoredAssetDialogIssue(
            identity,
            state.asset,
            tuple(candidate.path for candidate in state.candidates),
            state.source_directory,
        )
        self._state = replace(
            state,
            phase=AuthoredAssetPhase.CONFIRM_ISSUED,
            dialog_identity=identity,
        )
        return AuthoredAssetTransition(
            AuthoredAssetRefreshEffect.DIALOG, issue=issue
        )

    def present_confirmation(
        self, dialog: object, current_stamp: object
    ) -> AuthoredAssetTransition:
        state = self._state
        if (
            type(state) is not _OpenState
            or state.phase is not AuthoredAssetPhase.CONFIRM_ISSUED
            or dialog is not state.dialog_identity
        ):
            return AuthoredAssetTransition()
        if (
            not _context_current(state, current_stamp)
            or not all(_candidate_current(item) for item in state.candidates)
        ):
            return self._dismiss(
                state,
                dialog,
                "Authored asset context changed before presentation.",
            )
        self._state = replace(
            state, phase=AuthoredAssetPhase.CONFIRM_PRESENTED
        )
        return AuthoredAssetTransition(
            AuthoredAssetRefreshEffect.DIALOG,
            dialog=AuthoredAssetDialogCommand(
                dialog, AuthoredAssetDialogEffect.OPEN
            ),
        )

    def saved_mask_validation_request(
        self, current_stamp: object,
    ) -> AssetValidationRequest | None:
        state = self._state
        if (type(state) is not _OpenState or state.asset != "mask"
                or state.phase is not AuthoredAssetPhase.TERMINAL_READY):
            return None
        if not _context_current(state, current_stamp):
            self._state = _OpenState()
            return None
        candidate, = state.candidates
        return AssetValidationRequest(
            "mask", candidate.path, state.expected_shape,
            candidate, state.source_request,
        )

    def validation_request(
        self,
        dialog: object,
        path: object,
        current_stamp: object,
    ) -> AssetValidationRequest | None:
        state = self._state
        if (
            type(state) is not _OpenState
            or state.phase is not AuthoredAssetPhase.CONFIRM_PRESENTED
            or dialog is not state.dialog_identity
            or type(path) is not str
            or not os.path.isabs(path)
            or not _context_current(state, current_stamp)
            or state.asset is None
        ):
            return None
        candidate = next(
            (item for item in state.candidates if item.path == path), None
        )
        try:
            return AssetValidationRequest(
                state.asset,
                path,
                state.expected_shape,
                candidate,
                state.source_request,
            )
        except (TypeError, ValueError):
            return None

    def adopt_validation(
        self,
        dialog: object,
        request: object,
        identity: object,
    ) -> AuthoredAssetTransition:
        state = self._state
        if (
            type(state) is not _OpenState
            or not (state.phase is AuthoredAssetPhase.CONFIRM_PRESENTED
                    or state.asset == "mask"
                    and state.phase is AuthoredAssetPhase.TERMINAL_READY
                    and dialog is None)
            or dialog is not state.dialog_identity
            or type(request) is not AssetValidationRequest
            or type(identity) is not OperationIdentity
            or request.source_request is not state.source_request
            or request.asset != state.asset
            or request.expected_shape != state.expected_shape
        ):
            return AuthoredAssetTransition()
        try:
            request.__post_init__()
        except (TypeError, ValueError):
            return AuthoredAssetTransition()
        self._state = replace(
            state,
            phase=AuthoredAssetPhase.VALIDATING,
            operation_identity=identity,
            validation_request=request,
        )
        if dialog is None:
            return AuthoredAssetTransition(
                AuthoredAssetRefreshEffect.CONTROLS, "Validating saved mask…",
            )
        return AuthoredAssetTransition(
            AuthoredAssetRefreshEffect.DIALOG,
            "Validating the selected authored asset…",
            dialog=AuthoredAssetDialogCommand(
                dialog, AuthoredAssetDialogEffect.SET_BUSY
            ),
        )

    def _consume_validation_update(
        self,
        state: _OpenState,
        update: OperationUpdate,
        current_stamp: object,
    ) -> AuthoredAssetTransition:
        terminal = update.terminal
        if terminal is None:
            if update.progress is None:
                return AuthoredAssetTransition()
            return AuthoredAssetTransition(
                AuthoredAssetRefreshEffect.CONTROLS,
                "Validating the selected authored asset…",
            )
        dialog = state.dialog_identity
        if dialog is None and state.asset != "mask":
            self._state = _OpenState()
            return AuthoredAssetTransition(
                AuthoredAssetRefreshEffect.CONTROLS,
                "Authored asset was not adopted.",
            )
        if (
            update.stale
            or terminal.status is OperationTerminalStatus.CANCELLED
            or not _context_current(state, current_stamp)
        ):
            return self._dismiss(
                state,
                dialog,
                "Authored asset context changed; nothing was adopted.",
            )
        if terminal.status is not OperationTerminalStatus.RETURNED:
            if dialog is None:
                return replace(self._dismiss(
                    state, None, f"Mask validation failed: {terminal.diagnostic}"
                ), error=True)
            self._state = replace(
                state,
                phase=AuthoredAssetPhase.CONFIRM_PRESENTED,
                operation_identity=None,
                validation_request=None,
            )
            return AuthoredAssetTransition(
                AuthoredAssetRefreshEffect.DIALOG,
                f"Asset validation failed: {terminal.diagnostic}",
                dialog=AuthoredAssetDialogCommand(
                    dialog, AuthoredAssetDialogEffect.SET_IDLE
                ),
            )
        result = terminal.payload
        try:
            valid = (
                type(result) is AssetValidationResult
                and result.request is state.validation_request
                and result.candidate.asset == state.asset
                and result.candidate.path == result.request.path
            )
            if valid:
                terminal.__post_init__()
                result.__post_init__()
                valid = _candidate_current(result.candidate)
        except (AttributeError, TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            if dialog is None:
                return replace(self._dismiss(
                    state, None, "Mask validation returned an inexact result."
                ), error=True)
            self._state = replace(
                state,
                phase=AuthoredAssetPhase.CONFIRM_PRESENTED,
                operation_identity=None,
                validation_request=None,
            )
            return AuthoredAssetTransition(
                AuthoredAssetRefreshEffect.DIALOG,
                "Asset validation returned an inexact result.",
                dialog=AuthoredAssetDialogCommand(
                    dialog, AuthoredAssetDialogEffect.SET_IDLE
                ),
            )
        if (
            not _context_current(state, current_stamp)
            or not _candidate_current(result.candidate)
        ):
            return self._dismiss(
                state, dialog, "Authored asset changed before adoption."
            )
        before = self._intents.snapshot()
        control = PONI_FILE if state.asset == "poni" else MASK_FILE
        reduced = reduce_control_edit(
            before, control, result.candidate.path
        )
        if isinstance(reduced, (EditRefusal, EditNoChange)):
            notice = (
                f"Already selected: {result.candidate.path}"
                if isinstance(reduced, EditNoChange)
                else "The selected authored asset could not be adopted."
            )
            transition = self._dismiss(state, dialog, notice)
            if state.asset == "mask" and isinstance(reduced, EditNoChange):
                # The editor saved new contents to the already-selected path.
                transition = replace(transition, mask_set_path=result.candidate.path)
            return transition
        if (
            not _context_current(state, current_stamp)
            or not _candidate_current(result.candidate)
        ):
            return self._dismiss(
                state, dialog, "Authored asset changed before adoption."
            )
        try:
            committed = self._intents.commit(
                reduced, expected_revision=state.stamp.intent_revision
            )
        except Exception as error:
            return self._dismiss(
                state,
                dialog,
                f"Authored asset adoption failed: {error}",
            )
        if type(committed) not in {
            IntentCommitAccepted, IntentRecaptureRequired
        }:
            return self._dismiss(
                state,
                dialog,
                "Authored asset adoption returned invalid state.",
            )
        accepted = type(committed) is IntentCommitAccepted
        adoption = AuthoredAssetAdoption(
            before,
            committed,
            result.candidate.path,
            state.asset,
            accepted,
            dialog,
        )
        transition = self._dismiss(
            state,
            dialog,
            (
                f"{'PONI' if state.asset == 'poni' else 'Mask'} adopted: "
                f"{result.candidate.path}"
                if accepted
                else "Authored asset adoption was superseded."
            ),
        )
        return replace(
            transition, adoption=adoption,
            mask_set_path=(
                result.candidate.path if accepted and state.asset == "mask" else None
            ),
        )

    def _dismiss(
        self,
        state: _OpenState,
        dialog: AuthoredAssetDialogIdentity | None,
        notice: str,
    ) -> AuthoredAssetTransition:
        if dialog is None:
            self._state = _OpenState()
            return AuthoredAssetTransition(
                AuthoredAssetRefreshEffect.CONTROLS, notice,
            )
        self._state = replace(
            state,
            phase=AuthoredAssetPhase.DISMISSING,
            operation_identity=None,
            validation_request=None,
        )
        return AuthoredAssetTransition(
            AuthoredAssetRefreshEffect.DIALOG,
            notice,
            dialog=AuthoredAssetDialogCommand(
                dialog, AuthoredAssetDialogEffect.CLOSE
            ),
        )

    def cancel_confirmation(
        self, dialog: object
    ) -> AuthoredAssetTransition:
        state = self._state
        if (
            type(state) is not _OpenState
            or state.phase is not AuthoredAssetPhase.CONFIRM_PRESENTED
            or dialog is not state.dialog_identity
        ):
            return AuthoredAssetTransition()
        return self._dismiss(
            state,
            dialog,
            f"{'PONI' if state.asset == 'poni' else 'Mask'} was not adopted.",
        )

    def detach_dialog(
        self, dialog: object
    ) -> AuthoredAssetTransition:
        state = self._state
        if type(state) is _ClosingState:
            if (
                type(dialog) is not AuthoredAssetDialogIdentity
                or type(state.dialog_identity) is not AuthoredAssetDialogIdentity
                or dialog is not state.dialog_identity
            ):
                return AuthoredAssetTransition()
            replacement = replace(state, dialog_identity=None)
            self._state = replacement
            self._finish_close_if_ready(replacement)
            return AuthoredAssetTransition()
        if (
            type(dialog) is not AuthoredAssetDialogIdentity
            or type(state) is not _OpenState
            or type(state.dialog_identity) is not AuthoredAssetDialogIdentity
            or dialog is not state.dialog_identity
        ):
            return AuthoredAssetTransition()
        cancel = (
            state.operation_identity
            if state.phase is AuthoredAssetPhase.VALIDATING
            else None
        )
        if cancel is None:
            self._state = _OpenState()
        else:
            self._state = replace(
                state,
                phase=AuthoredAssetPhase.DISMISSING,
                dialog_identity=None,
            )
        return AuthoredAssetTransition(
            AuthoredAssetRefreshEffect.CONTROLS,
            "Authored asset was not adopted.",
            cancel_identity=cancel,
        )

    def operation_lost(
        self, identity: object
    ) -> AuthoredAssetTransition:
        state = self._state
        if type(state) is _ClosingState:
            if (
                type(identity) is not OperationIdentity
                or type(state.cleanup_identity) is not OperationIdentity
                or identity is not state.cleanup_identity
            ):
                return AuthoredAssetTransition()
            replacement = replace(
                state,
                cleanup_status=ClosingCleanupStatus.CLEANED,
                lost_owner_seen=True,
            )
            self._state = replacement
            self._finish_close_if_ready(replacement)
            return AuthoredAssetTransition()
        if (
            type(identity) is not OperationIdentity
            or type(state) is not _OpenState
            or type(state.operation_identity) is not OperationIdentity
            or identity is not state.operation_identity
        ):
            return AuthoredAssetTransition()
        dialog = state.dialog_identity
        if dialog is None:
            self._state = _OpenState()
            return AuthoredAssetTransition(
                AuthoredAssetRefreshEffect.CONTROLS,
                "Authored asset operation failed before terminal publication.",
            )
        return self._dismiss(
            state,
            dialog,
            "Authored asset operation failed before terminal publication.",
        )

    def begin_close(self) -> AuthoredAssetTransition:
        state = self._state
        if type(state) is not _OpenState:
            return AuthoredAssetTransition()
        cleanup_identity = state.operation_identity
        cleanup_status = (
            ClosingCleanupStatus.CLEANED
            if cleanup_identity is None
            else ClosingCleanupStatus.AWAITING
        )
        dialog = state.dialog_identity
        # Linearize the absorbing state before exposing cancellation or close.
        closing = _ClosingState(
            cleanup_identity, cleanup_status, dialog
        )
        self._state = closing
        self._finish_close_if_ready(closing)
        command = (
            None
            if dialog is None
            else AuthoredAssetDialogCommand(
                dialog, AuthoredAssetDialogEffect.CLOSE
            )
        )
        return AuthoredAssetTransition(
            AuthoredAssetRefreshEffect.DIALOG
            if command is not None
            else AuthoredAssetRefreshEffect.CONTROLS
            if cleanup_identity is not None
            else AuthoredAssetRefreshEffect.NONE,
            dialog=command,
            cancel_identity=cleanup_identity,
        )

    def consume_close_receipt(
        self, receipt: object
    ) -> AuthoredAssetTransition:
        if not _valid_detached_value(receipt, OperationCleanupReceipt):
            return AuthoredAssetTransition()
        state = self._state
        if (
            type(state) is not _ClosingState
            or state.cleanup_identity is None
            or receipt.identity is not state.cleanup_identity
        ):
            return AuthoredAssetTransition()
        if receipt.cleanup_status is CleanupStatus.CLEANUP_PENDING:
            if state.cleanup_status is ClosingCleanupStatus.CLEANED:
                return AuthoredAssetTransition()
            replacement = replace(
                state, cleanup_status=ClosingCleanupStatus.PENDING
            )
        elif receipt.cleanup_status is CleanupStatus.CLEANED:
            replacement = replace(
                state,
                cleanup_status=ClosingCleanupStatus.CLEANED,
                terminal_seen=receipt.terminal is not None,
            )
        else:
            return AuthoredAssetTransition()
        self._state = replacement
        self._finish_close_if_ready(replacement)
        return AuthoredAssetTransition()

    def _finish_close_if_ready(self, state: _ClosingState) -> None:
        if (
            self._state is state
            and state.cleanup_status is ClosingCleanupStatus.CLEANED
            and state.dialog_identity is None
        ):
            self._state = _ClosedState(
                state.cleanup_identity,
                state.terminal_seen,
                state.lost_owner_seen,
            )


__all__ = [
    "AuthoredAssetAdoption",
    "AuthoredAssetDialogCommand",
    "AuthoredAssetDialogEffect",
    "AuthoredAssetDialogIdentity",
    "AuthoredAssetDialogIssue",
    "AuthoredAssetEvidenceIdentity",
    "AuthoredAssetOwner",
    "AuthoredAssetOwnerLifecycle",
    "AuthoredAssetPhase",
    "AuthoredAssetRefreshEffect",
    "AuthoredAssetTransition",
    "ClosingCleanupStatus",
]
