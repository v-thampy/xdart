"""Hostile production-shaped Qt bridge oracles for authored assets."""

from __future__ import annotations

import os
from pathlib import Path
from threading import Event

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pyqtgraph.Qt import QtCore
import pytest

from xdart.gui.tabs.scattering import page as page_module
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.authored_assets import (
    AuthoredAssetDialogEffect,
    AuthoredAssetOwnerLifecycle,
    AuthoredAssetPhase,
    AuthoredAssetRefreshEffect,
    AuthoredAssetTransition,
)
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.events import CleanupStatus
from xdart.gui.tabs.scattering.experiment_authoring import (
    AssetValidationRequest,
)
from xdart.gui.tabs.scattering.operation_values import (
    OperationIdentity,
    OperationProgress,
    OperationTerminal,
    OperationTerminalStatus,
    OperationUpdate,
)
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xrd_tools.session.intent_store import (
    IntentCommitAccepted,
    IntentRecaptureRequired,
)

from tests.xdart.scattering.test_authored_asset_owner import (
    _Case,
    _case,
    _process_terminal,
    _validation_result,
)


@pytest.fixture
def qapp(_xdart_qt_harness):
    """Use the repository-owned QApplication instead of pytest-qt."""

    return _xdart_qt_harness.app


def _page(case: _Case) -> ScatteringWorkspace:
    page = ScatteringWorkspace(
        intents=case.store,
        lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter(),
    )
    # The fixture and page must share one exact owner and intent store.
    page._authored_assets = case.owner
    return page


def _adopt_running(
    page: ScatteringWorkspace,
    case: _Case,
    identity: OperationIdentity,
) -> None:
    transition = case.owner.adopt_operation(
        case.asset, case.request, case.stamp, identity,
    )
    assert transition.refresh is AuthoredAssetRefreshEffect.CONTROLS
    assert case.owner.phase is AuthoredAssetPhase.RUNNING
    assert page._authored_assets is case.owner


def _make_terminal_ready(
    page: ScatteringWorkspace,
    case: _Case,
    *, serial: int = 701,
) -> OperationIdentity:
    identity = OperationIdentity(serial)
    _adopt_running(page, case, identity)
    transition = case.owner.consume_operation_update(
        OperationUpdate(
            identity, terminal=_process_terminal(case, identity),
        ),
        case.stamp,
    )
    assert transition.refresh is AuthoredAssetRefreshEffect.CONTROLS
    assert case.owner.phase is AuthoredAssetPhase.TERMINAL_READY
    return identity


def _present_dialog(page: ScatteringWorkspace, case: _Case, qapp):
    evidence = case.owner.evidence_identity
    assert evidence is not None
    issued = case.owner.issue_confirmation(evidence, case.stamp)
    assert issued.issue is not None
    page._apply_authored_asset_transition(issued)
    dialog = page._authored_asset_dialog
    identity = page._authored_asset_dialog_identity
    assert dialog is not None
    assert identity is issued.issue.identity
    assert not dialog.isVisible()

    presented = case.owner.present_confirmation(identity, case.stamp)
    assert presented.dialog is not None
    assert presented.dialog.effect is AuthoredAssetDialogEffect.OPEN
    page._apply_authored_asset_transition(presented)
    qapp.processEvents()
    assert dialog.isVisible()
    assert case.owner.phase is AuthoredAssetPhase.CONFIRM_PRESENTED
    return dialog, identity


def _destroy_dialog(page: ScatteringWorkspace, qapp) -> None:
    dialog = getattr(page, "_authored_asset_dialog", None)
    if dialog is None:
        return
    dialog.close_inert()
    QtCore.QCoreApplication.sendPostedEvents(
        None, QtCore.QEvent.Type.DeferredDelete,
    )
    qapp.processEvents()


def _dispose(page: ScatteringWorkspace, qapp) -> None:
    page._run_timer.stop()
    _destroy_dialog(page, qapp)
    for _attempt in range(3):
        receipt = page.close_workspace()
        QtCore.QCoreApplication.sendPostedEvents(
            None, QtCore.QEvent.Type.DeferredDelete,
        )
        qapp.processEvents()
        if receipt.cleanup_status is CleanupStatus.CLEANED:
            break
    page.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(
        None, QtCore.QEvent.Type.DeferredDelete,
    )
    qapp.processEvents()


def _start_exact_slot_terminal(
    page: ScatteringWorkspace,
    case: _Case,
) -> OperationIdentity:
    slot = page._workspace_operations._slot
    identity = slot._begin(
        case.request,
        case.stamp,
        lambda _request, owned, _cancelled, _publish:
        _process_terminal(case, owned),
    )
    assert type(identity) is OperationIdentity
    _adopt_running(page, case, identity)
    worker = slot._worker
    assert worker is not None
    worker.join(3)
    assert not worker.is_alive()
    assert slot.current_identity is identity
    return identity


def test_real_timer_requires_two_ticks_to_issue_then_present_confirmation(
    tmp_path: Path, qapp, monkeypatch,
) -> None:
    case = _case(tmp_path)
    page = _page(case)
    refreshes: list[dict[str, object]] = []
    monkeypatch.setattr(
        page, "_refresh_shell",
        lambda **options: refreshes.append(dict(options)),
    )
    try:
        _start_exact_slot_terminal(page, case)
        page._ensure_timer()
        assert page._run_timer.isActive()

        page._run_timer.timeout.emit()
        assert case.owner.phase is AuthoredAssetPhase.CONFIRM_ISSUED
        dialog = page._authored_asset_dialog
        assert dialog is not None and not dialog.isVisible()
        assert page._run_timer.isActive()
        assert page._polling_needed()
        assert refreshes == [{
            "preserve_display": True,
            "preserve_scientific": True,
        }]

        page._run_timer.timeout.emit()
        qapp.processEvents()
        assert case.owner.phase is AuthoredAssetPhase.CONFIRM_PRESENTED
        assert page._authored_asset_dialog is dialog
        assert dialog.isVisible()
        assert not page._polling_needed()
        assert not page._run_timer.isActive()
        assert len(refreshes) == 1
    finally:
        _dispose(page, qapp)


def test_dialog_and_controls_effects_keep_exact_shell_boundaries(
    tmp_path: Path, qapp, monkeypatch,
) -> None:
    case = _case(tmp_path)
    page = _page(case)
    refreshes: list[dict[str, object]] = []
    browser_work: list[str] = []
    refresh_shell = page._refresh_shell
    reconcile = page._shell.browser.reconcile
    reconcile_residency = page._shell.browser.reconcile_heavy_residency

    def record_refresh(**options):
        refreshes.append(dict(options))
        return refresh_shell(**options)

    def record_reconcile(*args, **options):
        browser_work.append("reconcile")
        return reconcile(*args, **options)

    def record_residency(*args, **options):
        browser_work.append("residency")
        return reconcile_residency(*args, **options)

    def record_catalog_request(*_args, **_options):
        browser_work.append("catalog-request")
        raise AssertionError("authored effect requested a browser catalog")

    def record_catalog_poll(*_args, **_options):
        browser_work.append("catalog-poll")
        raise AssertionError("authored effect polled a browser catalog")

    monkeypatch.setattr(
        page, "_refresh_shell", record_refresh,
    )
    monkeypatch.setattr(page._shell.browser, "reconcile", record_reconcile)
    monkeypatch.setattr(
        page._shell.browser,
        "reconcile_heavy_residency",
        record_residency,
    )
    monkeypatch.setattr(
        page, "_request_browser_catalog", record_catalog_request,
    )
    monkeypatch.setattr(
        page._processed_browser, "request_catalog", record_catalog_request,
    )
    monkeypatch.setattr(
        page._processed_browser, "poll_catalog", record_catalog_poll,
    )
    try:
        identity = OperationIdentity(711)
        controls = case.owner.adopt_operation(
            case.asset, case.request, case.stamp, identity,
        )
        assert controls.refresh is AuthoredAssetRefreshEffect.CONTROLS
        page._apply_authored_asset_transition(controls)
        assert refreshes == [{
            "preserve_display": True,
            "preserve_scientific": True,
        }]
        assert browser_work == []

        refreshes.clear()
        ready = case.owner.consume_operation_update(
            OperationUpdate(
                identity, terminal=_process_terminal(case, identity),
            ),
            case.stamp,
        )
        assert ready.refresh is AuthoredAssetRefreshEffect.CONTROLS
        evidence = case.owner.evidence_identity
        assert evidence is not None
        issued = case.owner.issue_confirmation(evidence, case.stamp)
        assert issued.issue is not None
        page._apply_authored_asset_transition(issued)
        assert page._authored_asset_dialog is not None
        assert page._authored_asset_dialog_identity is issued.issue.identity
        assert refreshes == []
        assert browser_work == []
    finally:
        _dispose(page, qapp)


def test_progress_drain_performs_one_controls_reconcile_without_duplicate(
    tmp_path: Path, qapp, monkeypatch,
) -> None:
    case = _case(tmp_path)
    page = _page(case)
    slot = page._workspace_operations._slot
    entered, release = Event(), Event()
    refreshes: list[dict[str, object]] = []
    monkeypatch.setattr(
        page, "_refresh_shell",
        lambda **options: refreshes.append(dict(options)),
    )

    def held(_request, identity, _cancelled, publish):
        publish("discover", 1, 2)
        entered.set()
        assert release.wait(3)
        return _process_terminal(case, identity)

    try:
        identity = slot._begin(case.request, case.stamp, held)
        assert type(identity) is OperationIdentity
        _adopt_running(page, case, identity)
        assert entered.wait(2)

        page._drain_executor()
        assert case.owner.phase is AuthoredAssetPhase.RUNNING
        assert slot.current_identity is identity and slot.owned
        assert refreshes == [{
            "preserve_display": True,
            "preserve_scientific": True,
        }]
    finally:
        release.set()
        worker = slot._worker
        if worker is not None:
            worker.join(3)
        _dispose(page, qapp)


@pytest.mark.parametrize("recapture", (False, True))
def test_adoption_reconciles_accepted_and_recapture_but_remembers_only_accepted(
    tmp_path: Path, qapp, monkeypatch, recapture: bool,
) -> None:
    case = _case(tmp_path)
    page = _page(case)
    reconciled: list[tuple[object, object, dict[str, object]]] = []
    remembered: list[str] = []
    monkeypatch.setattr(page, "_refresh_shell", lambda **_options: None)
    try:
        _make_terminal_ready(page, case)
        _dialog, dialog_identity = _present_dialog(page, case, qapp)
        request = case.owner.validation_request(
            dialog_identity, case.candidate.path, case.stamp,
        )
        assert type(request) is AssetValidationRequest
        validation_identity = OperationIdentity(702)
        page._apply_authored_asset_transition(
            case.owner.adopt_validation(
                dialog_identity, request, validation_identity,
            )
        )
        if recapture:
            concurrent = case.store.snapshot().thaw()
            concurrent.project_root = str(tmp_path / "concurrent")
            assert type(case.store.commit(
                concurrent, expected_revision=case.store.revision,
            )) is IntentCommitAccepted

        terminal = OperationTerminal(
            validation_identity,
            OperationTerminalStatus.RETURNED,
            payload=_validation_result(request),
        )
        transition = case.owner.consume_operation_update(
            OperationUpdate(validation_identity, terminal=terminal),
            case.stamp,
        )
        adoption = transition.adoption
        assert adoption is not None
        expected_type = (
            IntentRecaptureRequired if recapture else IntentCommitAccepted
        )
        assert type(adoption.result) is expected_type
        assert adoption.remember_path is (not recapture)

        monkeypatch.setattr(
            page,
            "_reconcile_snapshot",
            lambda before, current, **options:
            reconciled.append((before, current, dict(options))),
        )
        monkeypatch.setattr(
            page_module, "remember_browse_path", remembered.append,
        )
        page._apply_authored_asset_transition(transition)

        assert len(reconciled) == 1
        before, current, options = reconciled[0]
        assert before is adoption.before
        assert current is adoption.result.snapshot
        assert options == {}
        assert remembered == ([] if recapture else [adoption.path])
    finally:
        _dispose(page, qapp)


def test_aborted_without_terminal_close_translates_exact_retirement_to_lost(
    tmp_path: Path, qapp, monkeypatch,
) -> None:
    case = _case(tmp_path)
    page = _page(case)
    slot = page._workspace_operations._slot
    lost: list[OperationIdentity] = []
    try:
        identity = slot._begin(
            case.request,
            case.stamp,
            lambda _request, _identity, _cancelled, _publish: object(),
        )
        assert type(identity) is OperationIdentity
        _adopt_running(page, case, identity)
        worker = slot._worker
        assert worker is not None
        worker.join(3)
        assert not worker.is_alive()
        assert slot._terminal is None
        assert slot._abort_fact is not None
        assert slot._abort_fact[0] == "ABORTED_WITHOUT_TERMINAL"
        assert slot.current_identity is identity and slot.owned

        operation_lost = case.owner.operation_lost

        def record_lost(candidate):
            if type(candidate) is OperationIdentity:
                lost.append(candidate)
            return operation_lost(candidate)

        monkeypatch.setattr(case.owner, "operation_lost", record_lost)
        receipt = page.close_workspace()
        slot_receipt = slot._clean_receipt
        assert slot_receipt is not None
        assert slot_receipt.cleanup_status is CleanupStatus.CLEANED
        assert slot_receipt.identity is None
        assert len(lost) == 1
        assert lost[0] is identity
        assert case.owner.lifecycle is AuthoredAssetOwnerLifecycle.CLOSED
        assert receipt.cleanup_status is CleanupStatus.CLEANED
    finally:
        _dispose(page, qapp)


def test_nonclosing_cancel_then_exact_returned_process_wins_to_confirmation(
    tmp_path: Path, qapp, monkeypatch,
) -> None:
    case = _case(tmp_path)
    page = _page(case)
    slot = page._workspace_operations._slot
    entered = Event()
    monkeypatch.setattr(page, "_refresh_shell", lambda **_options: None)

    def process_wins(_request, identity, cancelled, _publish):
        entered.set()
        assert cancelled.wait(3)
        return _process_terminal(case, identity)

    try:
        identity = slot._begin(case.request, case.stamp, process_wins)
        assert type(identity) is OperationIdentity
        _adopt_running(page, case, identity)
        assert entered.wait(2)
        assert not page._closing

        page._calibrate_action()
        assert not page._closing
        assert slot._cancel_event is not None
        assert slot._cancel_event.is_set()
        worker = slot._worker
        assert worker is not None
        worker.join(3)
        assert not worker.is_alive()
        assert slot._terminal is not None
        assert slot._terminal.status is OperationTerminalStatus.RETURNED

        page._ensure_timer()
        page._run_timer.timeout.emit()
        assert case.owner.phase is AuthoredAssetPhase.CONFIRM_ISSUED
        assert page._run_timer.isActive()
        page._run_timer.timeout.emit()
        qapp.processEvents()
        assert case.owner.phase is AuthoredAssetPhase.CONFIRM_PRESENTED
        dialog = page._authored_asset_dialog
        assert dialog is not None and dialog.isVisible()
    finally:
        _dispose(page, qapp)
