"""Qt-free ownership contract for deferred metadata operations."""

from __future__ import annotations

import subprocess
import sys
from typing import get_type_hints

from xdart.gui.tabs.scattering.metadata_operations import (
    MetadataDisposition,
    MetadataLifecycle,
    MetadataOperationOwner,
)
from xdart.gui.tabs.scattering.events import CleanupStatus
from xdart.gui.tabs.scattering.operation_values import (
    OperationCleanupReceipt,
    OperationContextStamp,
    OperationIdentity,
    OperationTerminal,
    OperationTerminalStatus,
)
from xrd_tools.analysis.scan_operations import MetadataTablePlan
from xrd_tools.analysis.scan_operations import (
    MetadataTableRequalificationPlan,
    MetadataTableResult,
)


def _request(
    owner: MetadataOperationOwner,
    target: str = "metadata",
    *,
    plan: object | None = None,
    facts: tuple[object, ...] | None = None,
    candidate: object | None = None,
):
    dialog = owner.dialog_identity(target) or owner.open_dialog(target)
    return owner.capture(
        MetadataTablePlan(f"/tmp/{target}.nexus") if plan is None else plan,
        ("metadata", target) if facts is None else facts,
        target=target,
        dialog=dialog,
        candidate=candidate,
    )


def test_metadata_owner_import_is_qt_free_in_fresh_interpreter() -> None:
    probe = (
        "import sys; "
        "import xdart.gui.tabs.scattering.metadata_operations; "
        "leaked=[name for name in sys.modules if name.split('.')[0] in "
        "('PySide6','PyQt5','PyQt6','qtpy','pyqtgraph')]; "
        "assert not leaked, leaked"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_owner_boundary_admits_only_exact_metadata_value_types() -> None:
    from xdart.gui.tabs.scattering.metadata_operations import MetadataRequest

    hints = get_type_hints(MetadataRequest)
    assert hints["plan"] == (
        MetadataTablePlan | MetadataTableRequalificationPlan
    )
    assert hints["candidate"] == (MetadataTableResult | None)

    owner = MetadataOperationOwner()
    dialog = owner.open_dialog("metadata")
    assert dialog is not None
    assert owner.capture(
        object(),
        ("metadata", "forged-plan"),
        target="metadata",
        dialog=dialog,
    ) is None
    assert owner.capture(
        MetadataTablePlan("/tmp/exact.nexus"),
        ["metadata", "mutable-request"],
        target="metadata",
        dialog=dialog,
    ) is None
    assert owner.capture(
        MetadataTablePlan("/tmp/exact.nexus"),
        ("metadata", "forged-candidate"),
        target="metadata",
        dialog=dialog,
        candidate=object(),
    ) is None
    assert owner.deferred is None


def test_dialog_tokens_are_monotonic_and_stale_close_cannot_retire_reopen() -> None:
    owner = MetadataOperationOwner()
    first = owner.open_dialog("metadata")
    assert owner.open_dialog("metadata") is first
    assert owner.close_dialog(first) is None

    reopened = owner.open_dialog("metadata")
    assert reopened is not first
    assert reopened.generation == first.generation + 1
    assert owner.close_dialog(first) is None
    assert owner.dialog_identity("metadata") is reopened


def test_capture_is_latest_only_across_both_metadata_dialogs() -> None:
    owner = MetadataOperationOwner()
    first = _request(owner, facts=("metadata", "first"))
    latest = _request(
        owner,
        "scan_roi",
        facts=("metadata", "latest"),
    )

    assert first is not None and latest is not None
    assert owner.deferred is latest
    assert owner.classify(
        first,
        current_request=first.request,
        blocked=False,
        start_allowed=True,
        closing=False,
    ) is MetadataDisposition.PERMANENT
    assert not owner.drop(first)


def test_transient_request_stays_exact_until_ready_start_transfers_it() -> None:
    owner = MetadataOperationOwner()
    request = _request(owner)
    assert request is not None
    admission = dict(
        current_request=request.request,
        start_allowed=True,
        closing=False,
    )
    assert owner.classify(
        request, blocked=True, **admission,
    ) is MetadataDisposition.TRANSIENT
    assert owner.deferred is request
    assert owner.classify(
        request, blocked=False, **admission,
    ) is MetadataDisposition.READY

    identity = OperationIdentity(11)
    context = OperationContextStamp(2)
    process = owner.start(request, identity, context)
    assert process is not None
    assert process.identity is identity
    assert process.request is request
    assert process.context is context
    assert owner.active is process
    assert owner.deferred is None
    assert owner.start(
        request, OperationIdentity(12), OperationContextStamp(2),
    ) is None


def test_deferred_currentness_uses_request_but_launch_owns_context_identity() -> None:
    owner = MetadataOperationOwner()
    request = _request(
        owner,
        facts=("metadata", "exact"),
    )
    assert request is not None
    base = dict(blocked=False, start_allowed=True, closing=False)
    assert owner.classify(
        request,
        current_request=request.request,
        **base,
    ) is MetadataDisposition.READY
    assert owner.classify(
        request,
        current_request=("metadata", "changed"),
        **base,
    ) is MetadataDisposition.PERMANENT
    # Deferred custody deliberately survives an unrelated intent revision; the
    # exact launch stamp is captured only by the successful start CAS.
    assert owner.classify(
        request,
        current_request=request.request,
        **base,
    ) is MetadataDisposition.READY
    assert owner.classify(
        request,
        current_request=request.request,
        blocked=False,
        start_allowed=True,
        closing=True,
    ) is MetadataDisposition.PERMANENT

    identity = OperationIdentity(20)
    context = OperationContextStamp(3, "context", 4)
    process = owner.start(request, identity, context)
    assert process is not None
    assert owner.process_is_current(
        process,
        current_request=request.request,
        current_context=context,
    )
    assert not owner.process_is_current(
        process,
        current_request=request.request,
        current_context=OperationContextStamp(4, "context", 4),
    )


def test_dialog_close_drops_only_its_deferred_request_and_returns_active_cancel() -> None:
    owner = MetadataOperationOwner()
    request = _request(owner)
    assert request is not None
    identity = OperationIdentity(21)
    process = owner.start(request, identity, OperationContextStamp(1))
    assert process is not None

    newer = _request(owner, "scan_roi", facts=("metadata", "newer"))
    assert newer is not None and owner.deferred is newer
    assert owner.close_dialog(process.request.dialog) is identity
    assert owner.active is process
    assert owner.deferred is newer
    assert owner.close_dialog(newer.dialog) is None
    assert owner.deferred is None


def test_only_exact_active_identity_can_finish_or_be_reported_lost() -> None:
    owner = MetadataOperationOwner()
    request = _request(owner)
    identity = OperationIdentity(31)
    process = owner.start(request, identity, OperationContextStamp(1))
    assert process is not None

    assert owner.finish(OperationIdentity(32)) is None
    assert owner.active is process
    assert owner.finish(identity) is process
    assert owner.active is None

    next_request = _request(owner)
    next_identity = OperationIdentity(33)
    next_process = owner.start(
        next_request, next_identity, OperationContextStamp(1),
    )
    assert next_process is not None
    assert owner.lost(next_identity) is next_process
    assert owner.active is None


def test_active_process_and_newer_deferred_request_remain_distinguishable() -> None:
    owner = MetadataOperationOwner()
    first = _request(owner, facts=("metadata", "first"))
    identity = OperationIdentity(41)
    process = owner.start(first, identity, OperationContextStamp(1))
    latest = _request(owner, facts=("metadata", "latest"))

    assert process is not None and latest is not None
    assert owner.active is process
    assert owner.deferred is latest
    assert owner.has_newer_request(process)
    assert not owner.has_newer_request(object())


def test_close_is_terminal_and_returns_only_the_exact_active_identity() -> None:
    owner = MetadataOperationOwner()
    request = _request(owner)
    identity = OperationIdentity(51)
    assert owner.start(
        request, identity, OperationContextStamp(1),
    ) is not None
    _request(owner, "scan_roi", facts=("metadata", "queued"))

    assert owner.begin_close() is identity
    assert owner.lifecycle is MetadataLifecycle.CLOSING
    assert owner.deferred is None
    assert owner.dialog_identity("metadata") is None
    assert owner.dialog_identity("scan_roi") is None
    assert owner.capture(
        object(), ("metadata", "closed"), target="metadata",
        dialog=owner.open_dialog("metadata"),
    ) is None

    pending = OperationCleanupReceipt(
        identity,
        CleanupStatus.CLEANUP_PENDING,
        cancel_accepted=True,
        worker_identity=1,
    )
    assert owner.consume_close_receipt(pending)
    assert owner.lifecycle is MetadataLifecycle.CLOSING
    wrong = OperationIdentity(52)
    assert not owner.consume_close_receipt(OperationCleanupReceipt(
        wrong,
        CleanupStatus.CLEANED,
        worker_identity=2,
        terminal=OperationTerminal(
            wrong, OperationTerminalStatus.CANCELLED,
        ),
    ))
    assert owner.lifecycle is MetadataLifecycle.CLOSING
    assert owner.consume_close_receipt(OperationCleanupReceipt(
        identity,
        CleanupStatus.CLEANED,
        cancel_accepted=True,
        worker_identity=1,
        terminal=OperationTerminal(
            identity, OperationTerminalStatus.CANCELLED,
        ),
    ))
    assert owner.lifecycle is MetadataLifecycle.CLOSED
    assert owner.active is None


def test_closing_exact_lost_owner_settles_without_a_forged_receipt() -> None:
    owner = MetadataOperationOwner()
    request = _request(owner)
    identity = OperationIdentity(61)
    process = owner.start(request, identity, OperationContextStamp(1))
    assert process is not None
    assert owner.begin_close() is identity
    assert owner.lost(OperationIdentity(62)) is None
    assert owner.lifecycle is MetadataLifecycle.CLOSING
    assert owner.lost(identity) is process
    assert owner.lifecycle is MetadataLifecycle.CLOSED
