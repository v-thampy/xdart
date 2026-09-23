"""Finite hostile, Qt-free oracles for the authored-asset outer FSM."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import hashlib
from pathlib import Path

from fabio.edfimage import EdfImage
import numpy as np
import pytest
import tifffile

from xdart.gui.tabs.scattering.authored_assets import (
    AuthoredAssetDialogEffect,
    AuthoredAssetDialogCommand,
    AuthoredAssetDialogIdentity,
    AuthoredAssetEvidenceIdentity,
    AuthoredAssetOwner,
    AuthoredAssetOwnerLifecycle,
    AuthoredAssetPhase,
    AuthoredAssetRefreshEffect,
    AuthoredAssetTransition,
    ClosingCleanupStatus,
)
from xdart.gui.tabs.scattering.contracts import SourceFileState
from xdart.gui.tabs.scattering.events import CleanupStatus
from xdart.gui.tabs.scattering.experiment_authoring import (
    AssetValidationRequest,
    AssetValidationResult,
    AuthoredAssetCandidate,
    CalibrationCandidate,
    CalibrationFileProof,
    CalibrationRequest,
    CalibrationResult,
    MaskProof,
    MaskRequest,
    MaskResult,
)
from xdart.gui.tabs.scattering.operation_values import (
    OperationCleanupReceipt,
    OperationContextStamp,
    OperationIdentity,
    OperationPending,
    OperationProgress,
    OperationTerminal,
    OperationTerminalStatus,
    OperationUpdate,
)
from xrd_tools.session.intent_store import (
    IntentCommitAccepted,
    IntentRecaptureRequired,
    RunIntentStore,
)
from xrd_tools.session.run_configuration import RunIntent


@dataclass(frozen=True)
class _Case:
    owner: AuthoredAssetOwner
    store: RunIntentStore
    stamp: OperationContextStamp
    asset: str
    request: CalibrationRequest | MaskRequest
    candidate: AuthoredAssetCandidate
    result: CalibrationResult | MaskResult


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case(tmp_path: Path, asset: str = "poni") -> _Case:
    root = tmp_path / asset
    root.mkdir(parents=True)
    source = root / "source.tiff"
    tifffile.imwrite(source, np.ones((1, 1), dtype=np.uint16))
    store = RunIntentStore(RunIntent(project_root=str(root)))
    stamp = OperationContextStamp(store.revision)
    if asset == "poni":
        executable = root / "pyFAI-calib2"
        executable.write_bytes(b"binary")
        executable.chmod(0o700)
        candidate_path = root / "made.poni"
        candidate_path.write_text("qualified", encoding="utf-8")
        candidate_state = SourceFileState.capture(candidate_path)
        proof = CalibrationFileProof(
            candidate_state,
            _sha(candidate_path),
            "{}",
            (0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0e-10),
        )
        request = CalibrationRequest(
            str(source),
            str(executable),
            SourceFileState.capture(executable),
            None,
            SourceFileState.capture(source),
            str(root),
            (root.stat().st_dev, root.stat().st_ino),
        )
        candidate = AuthoredAssetCandidate(
            "poni", str(candidate_path), proof, candidate_state,
        )
        result = CalibrationResult(
            request,
            (CalibrationCandidate(candidate.path, proof),),
            0,
            (str(executable), str(source)),
            str(root),
        )
    elif asset == "mask":
        executable = root / "pyFAI-drawmask"
        executable.write_bytes(b"binary")
        executable.chmod(0o700)
        final = root / "source-mask.edf"
        EdfImage(data=np.ones((1, 1), dtype=np.uint8)).write(str(final))
        final.chmod(0o600)
        request = MaskRequest(
            str(source),
            str(final),
            str(executable),
            SourceFileState.capture(executable),
            SourceFileState.capture(source),
            str(root),
            (root.stat().st_dev, root.stat().st_ino),
        )
        state = SourceFileState.capture(final)
        source_sha = _sha(source)
        proof = MaskProof(
            state,
            (1, 1),
            np.dtype("u2").str,
            np.dtype("u1").str,
            source_sha,
            source_sha,
            _sha(final),
            "zero-false-real-nonzero-true-nan-true-v1",
        )
        candidate = AuthoredAssetCandidate(
            "mask", str(final), proof, state, str(source),
        )
        result = MaskResult(
            request,
            str(final),
            proof,
            state,
            0,
            (str(executable), str(source)),
            str(root),
            True,
        )
    else:  # pragma: no cover - helper contract
        raise AssertionError(asset)
    return _Case(
        AuthoredAssetOwner(store), store, stamp, asset,
        request, candidate, result,
    )


def _running(
    tmp_path: Path, asset: str = "poni", *, serial: int = 1,
) -> tuple[_Case, OperationIdentity]:
    case = _case(tmp_path, asset)
    identity = OperationIdentity(serial)
    transition = case.owner.adopt_operation(
        asset, case.request, case.stamp, identity,
    )
    assert transition.refresh is AuthoredAssetRefreshEffect.CONTROLS
    assert case.owner.phase is AuthoredAssetPhase.RUNNING
    return case, identity


def _process_terminal(
    case: _Case, identity: OperationIdentity,
) -> OperationTerminal:
    return OperationTerminal(
        identity, OperationTerminalStatus.RETURNED, payload=case.result,
    )


def _terminal_ready(
    tmp_path: Path, asset: str = "poni", *, serial: int = 1,
) -> tuple[_Case, OperationIdentity, AuthoredAssetEvidenceIdentity]:
    case, identity = _running(tmp_path, asset, serial=serial)
    transition = case.owner.consume_operation_update(
        OperationUpdate(
            identity, terminal=_process_terminal(case, identity),
        ),
        case.stamp,
    )
    assert transition.refresh is AuthoredAssetRefreshEffect.CONTROLS
    assert transition.issue is transition.dialog is transition.adoption is None
    assert case.owner.phase is AuthoredAssetPhase.TERMINAL_READY
    evidence = case.owner.evidence_identity
    assert type(evidence) is AuthoredAssetEvidenceIdentity
    return case, identity, evidence


def _presented(
    tmp_path: Path, asset: str = "poni", *, serial: int = 1,
) -> tuple[_Case, AuthoredAssetDialogIdentity]:
    case, _identity, evidence = _terminal_ready(
        tmp_path, asset, serial=serial,
    )
    issued = case.owner.issue_confirmation(evidence, case.stamp)
    assert issued.issue is not None
    dialog = issued.issue.identity
    assert case.owner.phase is AuthoredAssetPhase.CONFIRM_ISSUED
    opened = case.owner.present_confirmation(dialog, case.stamp)
    assert opened.dialog is not None
    assert opened.dialog.effect is AuthoredAssetDialogEffect.OPEN
    assert opened.dialog.identity is dialog
    assert case.owner.phase is AuthoredAssetPhase.CONFIRM_PRESENTED
    return case, dialog


def _validating(
    tmp_path: Path, asset: str = "poni", *, serial: int = 1,
) -> tuple[
    _Case, AuthoredAssetDialogIdentity, AssetValidationRequest,
    OperationIdentity,
]:
    case, dialog = _presented(tmp_path, asset, serial=serial)
    request = case.owner.validation_request(
        dialog, case.candidate.path, case.stamp,
    )
    assert type(request) is AssetValidationRequest
    identity = OperationIdentity(serial + 100)
    adopted = case.owner.adopt_validation(dialog, request, identity)
    assert adopted.dialog is not None
    assert adopted.dialog.effect is AuthoredAssetDialogEffect.SET_BUSY
    assert case.owner.phase is AuthoredAssetPhase.VALIDATING
    return case, dialog, request, identity


def _clean_receipt(
    identity: OperationIdentity,
    terminal: OperationTerminal,
) -> OperationCleanupReceipt:
    assert terminal.identity is identity
    return OperationCleanupReceipt(
        identity,
        CleanupStatus.CLEANED,
        True,
        991,
        terminal,
    )


def _validation_result(
    request: AssetValidationRequest,
) -> AssetValidationResult:
    candidate = request.candidate
    assert type(candidate) is AuthoredAssetCandidate
    return AssetValidationResult(request, candidate)


def _forge(value: object, **changes: object) -> object:
    forged = object.__new__(type(value))
    for field in fields(value):
        object.__setattr__(
            forged,
            field.name,
            changes.get(field.name, getattr(value, field.name)),
        )
    return forged


@pytest.mark.parametrize("asset", ("poni", "mask"))
def test_process_wins_cancel_but_closing_never_issues_confirmation(
    tmp_path: Path, asset: str,
) -> None:
    case, identity = _running(tmp_path, asset)
    before = case.store.snapshot()
    closing = case.owner.begin_close()
    assert closing.cancel_identity is identity
    assert case.owner.lifecycle is AuthoredAssetOwnerLifecycle.CLOSING

    terminal = _process_terminal(case, identity)
    returned = case.owner.consume_operation_update(
        OperationUpdate(identity, terminal=terminal), case.stamp,
    )
    assert returned == AuthoredAssetTransition()
    assert case.owner.evidence_identity is None
    assert case.owner.lifecycle is AuthoredAssetOwnerLifecycle.CLOSING
    assert case.store.snapshot() == before

    case.owner.consume_close_receipt(_clean_receipt(identity, terminal))
    assert case.owner.lifecycle is AuthoredAssetOwnerLifecycle.CLOSED
    assert case.store.snapshot() == before


def test_terminal_evidence_is_separate_exact_and_single_use(tmp_path: Path) -> None:
    case, _identity, evidence = _terminal_ready(tmp_path)
    clone = AuthoredAssetEvidenceIdentity(evidence.serial)
    assert clone == evidence and clone is not evidence
    assert case.owner.issue_confirmation(
        clone, case.stamp,
    ) == AuthoredAssetTransition()
    assert case.owner.phase is AuthoredAssetPhase.TERMINAL_READY

    issued = case.owner.issue_confirmation(evidence, case.stamp)
    assert issued.issue is not None
    assert issued.dialog is issued.adoption is None
    with pytest.raises(ValueError, match="transition"):
        replace(issued, cancel_identity=OperationIdentity(44))
    assert case.owner.issue_confirmation(
        evidence, case.stamp,
    ) == AuthoredAssetTransition()


def test_none_never_matches_an_absent_operation_or_dialog_identity(
    tmp_path: Path,
) -> None:
    running, identity = _running(tmp_path / "running")
    assert running.owner.detach_dialog(None) == AuthoredAssetTransition()
    assert running.owner.operation_lost(None) == AuthoredAssetTransition()
    assert running.owner.phase is AuthoredAssetPhase.RUNNING
    assert running.owner.operation_identity is identity

    ready, _identity, _evidence = _terminal_ready(tmp_path / "ready")
    assert ready.owner.detach_dialog(None) == AuthoredAssetTransition()
    assert ready.owner.operation_lost(None) == AuthoredAssetTransition()
    assert ready.owner.phase is AuthoredAssetPhase.TERMINAL_READY


@pytest.mark.parametrize(
    "boundary",
    (
        "issue-source", "issue-candidate", "issue-stamp",
        "present-source", "present-candidate", "present-stamp",
    ),
)
def test_confirmation_boundaries_recheck_context_and_candidate(
    tmp_path: Path, boundary: str,
) -> None:
    case, _identity, evidence = _terminal_ready(tmp_path)
    at_issue, kind = boundary.split("-", 1)
    if at_issue == "present":
        issued = case.owner.issue_confirmation(evidence, case.stamp)
        assert issued.issue is not None
        dialog = issued.issue.identity
    else:
        dialog = None
    stamp = case.stamp
    if kind == "source":
        source = Path(case.request.source_path)
        source.write_bytes(source.read_bytes() + b"drift")
    elif kind == "candidate":
        candidate = Path(case.candidate.path)
        candidate.write_bytes(candidate.read_bytes() + b"drift")
    else:
        stamp = OperationContextStamp(case.stamp.intent_revision + 1)
    if at_issue == "issue":
        transition = case.owner.issue_confirmation(evidence, stamp)
        assert transition.issue is transition.dialog is None
        assert case.owner.phase is AuthoredAssetPhase.IDLE
    else:
        transition = case.owner.present_confirmation(dialog, stamp)
        assert transition.dialog is not None
        assert transition.dialog.identity is dialog
        assert transition.dialog.effect is AuthoredAssetDialogEffect.CLOSE
        assert case.owner.phase is AuthoredAssetPhase.DISMISSING


def test_close_between_terminal_and_issuance_is_absorbing(tmp_path: Path) -> None:
    case, _identity, evidence = _terminal_ready(tmp_path)
    assert case.owner.begin_close().cancel_identity is None
    assert case.owner.lifecycle is AuthoredAssetOwnerLifecycle.CLOSED
    assert case.owner.issue_confirmation(
        evidence, case.stamp,
    ) == AuthoredAssetTransition()


def test_equal_dialog_identity_is_inert_before_exact_presentation_and_close(
    tmp_path: Path,
) -> None:
    case, _identity, evidence = _terminal_ready(tmp_path)
    issued = case.owner.issue_confirmation(evidence, case.stamp)
    dialog = issued.issue.identity
    clone = AuthoredAssetDialogIdentity(dialog.serial)
    assert clone == dialog and clone is not dialog
    assert case.owner.present_confirmation(
        clone, case.stamp,
    ) == AuthoredAssetTransition()
    assert case.owner.phase is AuthoredAssetPhase.CONFIRM_ISSUED
    assert case.owner.present_confirmation(dialog, case.stamp).dialog.effect \
        is AuthoredAssetDialogEffect.OPEN
    assert case.owner.cancel_confirmation(clone) == AuthoredAssetTransition()
    assert case.owner.phase is AuthoredAssetPhase.CONFIRM_PRESENTED


def test_close_from_presented_blocks_exact_validation_and_set_busy(
    tmp_path: Path,
) -> None:
    case, dialog = _presented(tmp_path)
    request = case.owner.validation_request(
        dialog, case.candidate.path, case.stamp,
    )
    assert type(request) is AssetValidationRequest
    closing = case.owner.begin_close()
    assert closing.dialog.identity is dialog
    assert closing.dialog.effect is AuthoredAssetDialogEffect.CLOSE
    assert case.owner.adopt_validation(
        dialog, request, OperationIdentity(91),
    ) == AuthoredAssetTransition()
    assert case.owner.lifecycle is AuthoredAssetOwnerLifecycle.CLOSING


def test_close_between_issuance_and_presentation_closes_exact_dialog(
    tmp_path: Path,
) -> None:
    case, _identity, evidence = _terminal_ready(tmp_path)
    issued = case.owner.issue_confirmation(evidence, case.stamp)
    dialog = issued.issue.identity
    closing = case.owner.begin_close()
    assert closing.dialog is not None
    assert closing.dialog.identity is dialog
    assert closing.dialog.effect is AuthoredAssetDialogEffect.CLOSE
    assert case.owner.lifecycle is AuthoredAssetOwnerLifecycle.CLOSING
    assert case.owner.present_confirmation(
        dialog, case.stamp,
    ) == AuthoredAssetTransition()
    case.owner.detach_dialog(dialog)
    assert case.owner.lifecycle is AuthoredAssetOwnerLifecycle.CLOSED


def test_dialog_destroyed_while_validating_cancels_exact_operation(
    tmp_path: Path,
) -> None:
    case, dialog, _request, identity = _validating(tmp_path)
    detached = case.owner.detach_dialog(dialog)
    assert detached.cancel_identity is identity
    assert case.owner.phase is AuthoredAssetPhase.DISMISSING
    assert case.owner.dialog_identity is None
    lost = case.owner.operation_lost(identity)
    assert lost.refresh is AuthoredAssetRefreshEffect.CONTROLS
    assert case.owner.phase is AuthoredAssetPhase.IDLE


@pytest.mark.parametrize("settlement", ("returned", "failed", "lost"))
def test_closing_pending_terminal_or_lost_cleans_frozen_target_without_cas(
    tmp_path: Path, settlement: str,
) -> None:
    case, dialog, request, identity = _validating(tmp_path)
    before = case.store.snapshot()
    closing = case.owner.begin_close()
    assert closing.cancel_identity is identity
    assert closing.dialog.identity is dialog
    assert case.owner.cleanup_identity is identity
    assert case.owner.closing_cleanup_status is ClosingCleanupStatus.AWAITING

    pending = OperationCleanupReceipt(
        identity, CleanupStatus.CLEANUP_PENDING, True, 41,
    )
    case.owner.consume_close_receipt(pending)
    assert case.owner.closing_cleanup_status is ClosingCleanupStatus.PENDING

    if settlement == "lost":
        transition = case.owner.operation_lost(identity)
        assert transition == AuthoredAssetTransition()
    else:
        terminal = OperationTerminal(
            identity,
            OperationTerminalStatus.RETURNED
            if settlement == "returned" else OperationTerminalStatus.FAILED,
            "" if settlement == "returned" else "validation failed",
            _validation_result(request)
            if settlement == "returned" else None,
        )
        transition = case.owner.consume_operation_update(
            OperationUpdate(identity, terminal=terminal), case.stamp,
        )
        assert transition == AuthoredAssetTransition()
        assert case.owner.closing_terminal_seen
        transition = case.owner.consume_close_receipt(
            _clean_receipt(identity, terminal),
        )
    assert transition.issue is transition.dialog is transition.adoption is None
    assert case.owner.cleanup_identity is identity
    assert case.owner.closing_cleanup_status is ClosingCleanupStatus.CLEANED
    assert case.owner.lifecycle is AuthoredAssetOwnerLifecycle.CLOSING
    assert case.store.snapshot() == before

    if settlement == "lost":
        terminal = OperationTerminal(
            identity, OperationTerminalStatus.CANCELLED,
        )
        assert case.owner.consume_close_receipt(
            _clean_receipt(identity, terminal),
        ) == AuthoredAssetTransition()
        assert case.owner.closing_cleanup_status \
            is ClosingCleanupStatus.CLEANED

    case.owner.detach_dialog(dialog)
    assert case.owner.lifecycle is AuthoredAssetOwnerLifecycle.CLOSED
    assert case.store.snapshot() == before


def test_idle_close_freezes_explicit_idle_clean_target_and_receipts_are_inert(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    first = case.owner.begin_close()
    assert first == AuthoredAssetTransition()
    assert case.owner.lifecycle is AuthoredAssetOwnerLifecycle.CLOSED
    assert case.owner.cleanup_identity is None
    assert case.owner.closing_cleanup_status is ClosingCleanupStatus.CLEANED
    foreign = OperationIdentity(8)
    foreign_terminal = OperationTerminal(
        foreign, OperationTerminalStatus.CANCELLED,
    )
    assert case.owner.consume_close_receipt(
        _clean_receipt(foreign, foreign_terminal),
    ) == AuthoredAssetTransition()
    assert case.owner.operation_lost(foreign) == AuthoredAssetTransition()
    assert case.owner.begin_close() == AuthoredAssetTransition()


def test_only_exact_cleanup_receipt_can_clean_active_frozen_target(
    tmp_path: Path,
) -> None:
    case, identity = _running(tmp_path)
    case.owner.begin_close()
    idle = OperationCleanupReceipt(None, CleanupStatus.CLEANED)
    assert case.owner.consume_close_receipt(idle) == AuthoredAssetTransition()
    unequal = OperationIdentity(identity.serial + 1)
    unequal_terminal = OperationTerminal(
        unequal, OperationTerminalStatus.CANCELLED,
    )
    case.owner.consume_close_receipt(_clean_receipt(unequal, unequal_terminal))
    equal = OperationIdentity(identity.serial)
    equal_terminal = OperationTerminal(
        equal, OperationTerminalStatus.CANCELLED,
    )
    case.owner.consume_close_receipt(_clean_receipt(equal, equal_terminal))
    assert case.owner.cleanup_identity is identity
    assert case.owner.closing_cleanup_status is ClosingCleanupStatus.AWAITING
    exact_terminal = OperationTerminal(
        identity, OperationTerminalStatus.CANCELLED,
    )
    case.owner.consume_close_receipt(_clean_receipt(identity, exact_terminal))
    assert case.owner.lifecycle is AuthoredAssetOwnerLifecycle.CLOSED


def test_active_equal_valued_foreign_operation_events_are_inert(
    tmp_path: Path,
) -> None:
    case, identity = _running(tmp_path)
    clone = OperationIdentity(identity.serial)
    assert clone == identity and clone is not identity
    progress = OperationProgress(clone, "discover", 1, 2, 1)
    assert case.owner.consume_operation_update(
        OperationUpdate(clone, progress=progress), case.stamp,
    ) == AuthoredAssetTransition()
    terminal = OperationTerminal(
        clone, OperationTerminalStatus.RETURNED, payload=case.result,
    )
    assert case.owner.consume_operation_update(
        OperationUpdate(clone, terminal=terminal), case.stamp,
    ) == AuthoredAssetTransition()
    assert case.owner.operation_lost(clone) == AuthoredAssetTransition()
    assert case.owner.phase is AuthoredAssetPhase.RUNNING
    assert case.owner.operation_identity is identity


@pytest.mark.parametrize("nested_kind", ("progress", "pending", "terminal"))
@pytest.mark.parametrize("identity_kind", ("equal", "foreign"))
def test_forged_exact_outer_update_rejects_every_inexact_nested_identity(
    tmp_path: Path, nested_kind: str, identity_kind: str,
) -> None:
    case, identity = _running(tmp_path)
    nested_identity = OperationIdentity(
        identity.serial
        if identity_kind == "equal"
        else identity.serial + 1_000
    )
    assert nested_identity is not identity
    if identity_kind == "equal":
        assert nested_identity == identity
    else:
        assert nested_identity != identity

    exact_progress = OperationProgress(identity, "discover", 1, 2, 1)
    if nested_kind == "progress":
        forged = _forge(
            OperationUpdate(identity, progress=exact_progress),
            progress=OperationProgress(
                nested_identity, "discover", 1, 2, 1,
            ),
        )
    elif nested_kind == "pending":
        # The exact progress is a side-effect sentinel: an owner that ignores
        # the invalid nested pending fact would emit a controls transition.
        forged = _forge(
            OperationUpdate(identity, progress=exact_progress),
            pending=OperationPending(
                nested_identity, 1, "awaiting-choice", "choose one",
            ),
        )
    else:
        forged = _forge(
            OperationUpdate(
                identity, terminal=_process_terminal(case, identity),
            ),
            terminal=OperationTerminal(
                nested_identity,
                OperationTerminalStatus.RETURNED,
                payload=case.result,
            ),
        )
    assert type(forged) is OperationUpdate
    assert forged.identity is identity
    with pytest.raises(ValueError, match="operation update"):
        forged.__post_init__()

    before = case.store.snapshot()
    assert case.owner.consume_operation_update(
        forged, case.stamp,
    ) == AuthoredAssetTransition()
    assert case.owner.phase is AuthoredAssetPhase.RUNNING
    assert case.owner.operation_identity is identity
    assert case.store.snapshot() == before


@pytest.mark.parametrize("identity_kind", ("equal", "foreign"))
def test_forged_validation_terminal_cannot_reach_intent_cas(
    tmp_path: Path, identity_kind: str,
) -> None:
    case, _dialog, request, identity = _validating(tmp_path)
    nested_identity = OperationIdentity(
        identity.serial
        if identity_kind == "equal"
        else identity.serial + 1_000
    )
    exact_terminal = OperationTerminal(
        identity,
        OperationTerminalStatus.RETURNED,
        payload=_validation_result(request),
    )
    forged = _forge(
        OperationUpdate(identity, terminal=exact_terminal),
        terminal=OperationTerminal(
            nested_identity,
            OperationTerminalStatus.RETURNED,
            payload=_validation_result(request),
        ),
    )
    assert type(forged) is OperationUpdate
    assert forged.identity is identity
    with pytest.raises(ValueError, match="operation update"):
        forged.__post_init__()

    before = case.store.snapshot()
    assert case.owner.consume_operation_update(
        forged, case.stamp,
    ) == AuthoredAssetTransition()
    assert case.owner.phase is AuthoredAssetPhase.VALIDATING
    assert case.owner.operation_identity is identity
    assert case.store.snapshot() == before


@pytest.mark.parametrize("identity_kind", ("equal", "foreign"))
def test_closing_revalidates_forged_update_before_recording_terminal_evidence(
    tmp_path: Path, identity_kind: str,
) -> None:
    case, identity = _running(tmp_path)
    before = case.store.snapshot()
    closing = case.owner.begin_close()
    assert closing.cancel_identity is identity
    assert case.owner.lifecycle is AuthoredAssetOwnerLifecycle.CLOSING
    assert case.owner.cleanup_identity is identity
    assert case.owner.closing_cleanup_status is ClosingCleanupStatus.AWAITING
    assert not case.owner.closing_terminal_seen

    nested_identity = OperationIdentity(
        identity.serial
        if identity_kind == "equal"
        else identity.serial + 1_000
    )
    assert nested_identity is not identity
    if identity_kind == "equal":
        assert nested_identity == identity
    else:
        assert nested_identity != identity
    exact = OperationUpdate(
        identity, terminal=_process_terminal(case, identity),
    )
    forged = _forge(
        exact,
        terminal=OperationTerminal(
            nested_identity,
            OperationTerminalStatus.RETURNED,
            payload=case.result,
        ),
    )
    assert type(forged) is OperationUpdate
    assert forged.identity is identity
    with pytest.raises(ValueError, match="operation update"):
        forged.__post_init__()

    assert case.owner.consume_operation_update(
        forged, case.stamp,
    ) == AuthoredAssetTransition()
    assert case.owner.lifecycle is AuthoredAssetOwnerLifecycle.CLOSING
    assert case.owner.cleanup_identity is identity
    assert case.owner.closing_cleanup_status is ClosingCleanupStatus.AWAITING
    assert not case.owner.closing_terminal_seen
    assert case.owner.evidence_identity is None
    assert case.store.snapshot() == before


@pytest.mark.parametrize("terminal_kind", ("missing", "equal", "foreign"))
def test_forged_cleanup_receipt_cannot_clean_the_frozen_target(
    tmp_path: Path, terminal_kind: str,
) -> None:
    case, identity = _running(tmp_path)
    case.owner.begin_close()
    exact_terminal = OperationTerminal(
        identity, OperationTerminalStatus.CANCELLED,
    )
    receipt = _clean_receipt(identity, exact_terminal)
    if terminal_kind == "missing":
        forged = _forge(receipt, terminal=None)
    else:
        nested_identity = OperationIdentity(
            identity.serial
            if terminal_kind == "equal"
            else identity.serial + 1_000
        )
        forged = _forge(
            receipt,
            terminal=OperationTerminal(
                nested_identity, OperationTerminalStatus.CANCELLED,
            ),
        )
    assert type(forged) is OperationCleanupReceipt
    assert forged.identity is identity
    with pytest.raises(ValueError, match="operation cleanup receipt"):
        forged.__post_init__()

    assert case.owner.consume_close_receipt(
        forged,
    ) == AuthoredAssetTransition()
    assert case.owner.lifecycle is AuthoredAssetOwnerLifecycle.CLOSING
    assert case.owner.cleanup_identity is identity
    assert case.owner.closing_cleanup_status is ClosingCleanupStatus.AWAITING
    assert not case.owner.closing_terminal_seen


def test_late_progress_during_closing_never_emits_controls_or_dialog(
    tmp_path: Path,
) -> None:
    case, identity = _running(tmp_path)
    case.owner.begin_close()
    progress = OperationProgress(identity, "discover", 1, 2, 1)
    assert case.owner.consume_operation_update(
        OperationUpdate(identity, progress=progress), case.stamp,
    ) == AuthoredAssetTransition()


@pytest.mark.parametrize("asset", ("poni", "mask"))
def test_exact_validation_accepts_one_cas_and_duplicate_terminal_is_inert(
    tmp_path: Path, asset: str,
) -> None:
    if asset == "mask":
        case, _process, _evidence = _terminal_ready(tmp_path, asset)
        dialog = None
        request = case.owner.saved_asset_validation_request(case.stamp)
        identity = OperationIdentity(2)
        case.owner.adopt_validation(None, request, identity)
    else:
        case, dialog, request, identity = _validating(tmp_path, asset)
    terminal = OperationTerminal(
        identity,
        OperationTerminalStatus.RETURNED,
        payload=_validation_result(request),
    )
    transition = case.owner.consume_operation_update(
        OperationUpdate(identity, terminal=terminal), case.stamp,
    )
    if asset == "mask":
        assert transition.dialog is None
        assert transition.refresh is AuthoredAssetRefreshEffect.CONTROLS
    else:
        assert transition.dialog is not None
        assert transition.dialog.identity is dialog
        assert transition.dialog.effect is AuthoredAssetDialogEffect.CLOSE
    assert transition.adoption is not None
    assert type(transition.adoption.result) is IntentCommitAccepted
    assert transition.adoption.path == case.candidate.path
    assert transition.adoption.remember_path
    intent = case.store.snapshot().thaw()
    selected = intent.poni_file if asset == "poni" else intent.mask_file
    assert selected == case.candidate.path
    assert case.owner.phase is (AuthoredAssetPhase.IDLE if asset == "mask"
                                else AuthoredAssetPhase.DISMISSING)
    if asset == "poni":
        with pytest.raises(ValueError, match="transition"):
            replace(transition, dialog=AuthoredAssetDialogCommand(
                dialog, AuthoredAssetDialogEffect.SET_IDLE))
    assert case.owner.consume_operation_update(
        OperationUpdate(identity, terminal=terminal), case.stamp,
    ) == AuthoredAssetTransition()


def test_cas_race_is_superseded_without_remember_or_adoption(tmp_path: Path) -> None:
    case, dialog, request, identity = _validating(tmp_path)
    concurrent = case.store.snapshot().thaw()
    concurrent.project_root = str(tmp_path / "concurrent")
    assert type(case.store.commit(
        concurrent, expected_revision=case.store.revision,
    )) is IntentCommitAccepted
    transition = case.owner.consume_operation_update(
        OperationUpdate(
            identity,
            terminal=OperationTerminal(
                identity,
                OperationTerminalStatus.RETURNED,
                payload=_validation_result(request),
            ),
        ),
        case.stamp,
    )
    assert transition.dialog.identity is dialog
    assert transition.dialog.effect is AuthoredAssetDialogEffect.CLOSE
    assert transition.adoption is not None
    assert type(transition.adoption.result) is IntentRecaptureRequired
    assert not transition.adoption.remember_path
    assert case.store.snapshot().thaw().poni_file == ""


def test_current_stamp_mismatch_refuses_before_cas(tmp_path: Path) -> None:
    case, _dialog, request, identity = _validating(tmp_path)
    before = case.store.snapshot()
    foreign_stamp = OperationContextStamp(case.stamp.intent_revision + 1)
    transition = case.owner.consume_operation_update(
        OperationUpdate(
            identity,
            terminal=OperationTerminal(
                identity,
                OperationTerminalStatus.RETURNED,
                payload=_validation_result(request),
            ),
        ),
        foreign_stamp,
    )
    assert transition.adoption is None
    assert case.store.snapshot() == before


@pytest.mark.parametrize("mutation", ("candidate", "source"))
def test_candidate_or_source_mutation_refuses_before_cas(
    tmp_path: Path, mutation: str,
) -> None:
    case, _dialog, request, identity = _validating(tmp_path)
    before = case.store.snapshot()
    path = Path(
        case.candidate.path
        if mutation == "candidate" else case.request.source_path
    )
    path.write_bytes(path.read_bytes() + b"drift")
    transition = case.owner.consume_operation_update(
        OperationUpdate(
            identity,
            terminal=OperationTerminal(
                identity,
                OperationTerminalStatus.RETURNED,
                payload=_validation_result(request),
            ),
        ),
        case.stamp,
    )
    assert transition.adoption is None
    assert case.store.snapshot() == before


def test_close_before_validation_terminal_blocks_cas_and_set_idle(
    tmp_path: Path,
) -> None:
    case, dialog, request, identity = _validating(tmp_path)
    before = case.store.snapshot()
    closing = case.owner.begin_close()
    assert closing.dialog.identity is dialog
    terminal = OperationTerminal(
        identity,
        OperationTerminalStatus.RETURNED,
        payload=_validation_result(request),
    )
    assert case.owner.consume_operation_update(
        OperationUpdate(identity, terminal=terminal), case.stamp,
    ) == AuthoredAssetTransition()
    assert case.store.snapshot() == before
    case.owner.consume_close_receipt(_clean_receipt(identity, terminal))
    assert case.owner.lifecycle is AuthoredAssetOwnerLifecycle.CLOSING
    case.owner.detach_dialog(dialog)
    assert case.owner.lifecycle is AuthoredAssetOwnerLifecycle.CLOSED


def test_cleanup_may_finish_before_dialog_destruction_without_early_close(
    tmp_path: Path,
) -> None:
    case, dialog, _request, identity = _validating(tmp_path)
    case.owner.begin_close()
    terminal = OperationTerminal(
        identity, OperationTerminalStatus.CANCELLED,
    )
    case.owner.consume_close_receipt(_clean_receipt(identity, terminal))
    assert case.owner.lifecycle is AuthoredAssetOwnerLifecycle.CLOSING
    assert case.owner.closing_cleanup_status is ClosingCleanupStatus.CLEANED
    late_pending = OperationCleanupReceipt(
        identity, CleanupStatus.CLEANUP_PENDING, True, 991,
    )
    assert case.owner.consume_close_receipt(
        late_pending,
    ) == AuthoredAssetTransition()
    assert case.owner.closing_cleanup_status is ClosingCleanupStatus.CLEANED
    case.owner.detach_dialog(dialog)
    assert case.owner.lifecycle is AuthoredAssetOwnerLifecycle.CLOSED
