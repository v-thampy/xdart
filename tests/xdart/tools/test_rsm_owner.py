from __future__ import annotations

from dataclasses import replace
from threading import Event
import time

import pytest

from xdart.gui.pages.operation_owner import OperationTerminalStatus
from xdart.gui.pages.values import PageCleanup
from xdart.gui.tools.rsm_owner import (
    RSMOwnerAction,
    RSMOwnerFinalization,
    RSMOwnerOutcomeKind,
    RSMToolOwner,
)
from xdart.gui.tools.rsm_values import RSMToolPreflightRefused
from xrd_tools.analysis.module_transaction import (
    ModuleDisposition,
    ModuleProgress,
    ModuleSourceReceipt,
    ModuleTerminalResult,
)
from xrd_tools.analysis.rsm_operation import (
    RSMOperationCleanupPending,
    RSMOperationExecution,
    RSMOperationResult,
    RSMOperationVerificationError,
)


def _poll_terminal(owner: RSMToolOwner, *, timeout: float = 3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        update = owner.poll()
        if update is not None and update.terminal:
            return update
        time.sleep(0.001)
    raise AssertionError("RSM owner terminal did not arrive")


def _refused_result(execution: RSMOperationExecution) -> RSMOperationResult:
    return RSMOperationResult(
        execution.request,
        ModuleTerminalResult(
            execution.request.module,
            ModuleDisposition.REFUSED,
            "TEST_RESULT",
        ),
    )


def _prepare_owner(preflight):
    owner = RSMToolOwner(
        preflight_runner=lambda _form, *, cancel_token=None: preflight
    )
    assert owner.set_form(preflight.form) is False
    assert owner.begin_preflight() is not None
    update = _poll_terminal(owner)
    assert update.action is RSMOwnerAction.PREFLIGHT
    assert update.outcome.kind is RSMOwnerOutcomeKind.PREFLIGHT_READY
    assert owner.prepared is preflight
    return owner


def test_form_edit_and_bounce_cannot_resurrect_preflight(prepared_rsm_tool):
    preflight = prepared_rsm_tool
    owner = _prepare_owner(preflight)
    changed = replace(
        preflight.form,
        output_path=preflight.form.output_path.replace("rsm.nexus", "changed.nexus"),
    )
    assert owner.set_form(changed) is True
    assert owner.set_form(preflight.form) is False
    assert owner.prepared is None
    assert owner.begin_run() is None
    assert owner.close().status is PageCleanup.CLEAN


def test_inflight_preview_bounce_is_stale(prepared_rsm_tool):
    preflight = prepared_rsm_tool
    entered = Event()
    release = Event()

    def held(_form, *, cancel_token=None):
        entered.set()
        assert release.wait(2)
        return preflight

    owner = RSMToolOwner(preflight_runner=held)
    owner.set_form(preflight.form)
    assert owner.begin_preflight() is not None
    assert entered.wait(2)
    changed = replace(
        preflight.form,
        output_path=preflight.form.output_path.replace("rsm.nexus", "changed.nexus"),
    )
    owner.set_form(changed)
    owner.set_form(preflight.form)
    release.set()
    update = _poll_terminal(owner)
    assert update.stale is True
    assert update.outcome.kind is RSMOwnerOutcomeKind.PREFLIGHT_READY
    assert owner.prepared is None
    assert owner.close().status is PageCleanup.CLEAN


def test_failed_repreview_revokes_same_form_authority(prepared_rsm_tool):
    preflight = prepared_rsm_tool
    calls = 0

    def runner(_form, *, cancel_token=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            return preflight
        raise RuntimeError("source revision changed")

    owner = RSMToolOwner(preflight_runner=runner)
    owner.set_form(preflight.form)
    assert owner.begin_preflight() is not None
    assert _poll_terminal(owner).outcome.kind is RSMOwnerOutcomeKind.PREFLIGHT_READY
    assert owner.begin_preflight() is not None
    assert owner.prepared is None
    failed = _poll_terminal(owner)
    assert failed.terminal_status is OperationTerminalStatus.FAILED
    assert failed.failure_type == "RuntimeError"
    assert owner.begin_run() is None
    assert owner.close().status is PageCleanup.CLEAN


def test_preview_and_science_are_cooperatively_cancellable(
    monkeypatch,
    prepared_rsm_tool,
):
    preflight = prepared_rsm_tool
    preview_entered = Event()

    def held_preview(_form, *, cancel_token=None):
        preview_entered.set()
        assert cancel_token.wait(2)
        raise RSMToolPreflightRefused("CANCELLED")

    preview_owner = RSMToolOwner(preflight_runner=held_preview)
    preview_owner.set_form(preflight.form)
    assert preview_owner.begin_preflight() is not None
    assert preview_entered.wait(2)
    assert preview_owner.cancel() is True
    assert preview_owner.cancel() is False
    assert (
        _poll_terminal(preview_owner).terminal_status
        is OperationTerminalStatus.CANCELLED
    )
    assert preview_owner.close().status is PageCleanup.CLEAN

    science_entered = Event()

    def held_run(self, *, cancel_token=None, progress_callback=None):
        science_entered.set()
        assert cancel_token.wait(2)
        return _refused_result(self)

    monkeypatch.setattr(RSMOperationExecution, "run", held_run)
    owner = _prepare_owner(preflight)
    assert owner.begin_run() is not None
    assert science_entered.wait(2)
    assert owner.cancel() is True
    update = _poll_terminal(owner)
    assert update.outcome.kind is RSMOwnerOutcomeKind.RESULT
    assert owner.close().status is PageCleanup.CLEAN


def test_typed_preview_refusal_is_returned_without_prepared_authority(
    prepared_rsm_tool,
):
    preflight = prepared_rsm_tool

    def refused(_form, *, cancel_token=None):
        raise RSMToolPreflightRefused(
            "SOURCE_REVISION_CHANGED",
            "source changed during Preview",
            diagnostics=("primary SPEC revision changed",),
        )

    owner = RSMToolOwner(preflight_runner=refused)
    owner.set_form(preflight.form)
    assert owner.begin_preflight() is not None
    update = _poll_terminal(owner)

    assert update.terminal_status is OperationTerminalStatus.RETURNED
    assert update.outcome.kind is RSMOwnerOutcomeKind.PREFLIGHT_REFUSED
    assert update.outcome.refusal_code == "SOURCE_REVISION_CHANGED"
    assert "source changed during Preview" in update.outcome.refusal_message
    assert update.outcome.diagnostics == ("primary SPEC revision changed",)
    assert owner.prepared is None
    assert owner.begin_run() is None
    assert owner.close().status is PageCleanup.CLEAN


def test_production_preview_maps_exact_source_binding_drift_to_refusal(
    monkeypatch,
    rsm_tool_form,
):
    def drifted_source(cls, *_args, **_kwargs):
        raise ValueError("metadata table is no longer an exact source fact")

    monkeypatch.setattr(
        ModuleSourceReceipt,
        "from_metadata_table",
        classmethod(drifted_source),
    )
    owner = RSMToolOwner()
    owner.set_form(rsm_tool_form)
    assert owner.begin_preflight() is not None
    update = _poll_terminal(owner)

    assert update.terminal_status is OperationTerminalStatus.RETURNED
    assert update.outcome.kind is RSMOwnerOutcomeKind.PREFLIGHT_REFUSED
    assert update.outcome.refusal_code == "SOURCE_REVISION_CHANGED"
    assert "changed while binding" in update.outcome.refusal_message
    assert update.outcome.diagnostics == (
        "metadata table is no longer an exact source fact",
    )
    assert owner.prepared is None
    assert owner.begin_run() is None
    assert owner.close().status is PageCleanup.CLEAN


def test_latest_progress_and_cleanup_retry_do_not_replay(
    monkeypatch,
    prepared_rsm_tool,
):
    preflight = prepared_rsm_tool
    entered = Event()
    release = Event()
    calls = []

    def held_run(self, *, cancel_token=None, progress_callback=None):
        calls.append("run")
        progress_callback(ModuleProgress(self.request.module, 1, "science", 1, 3))
        progress_callback(ModuleProgress(self.request.module, 2, "science", 2, 3))
        entered.set()
        assert release.wait(2)
        raise RSMOperationCleanupPending(self)

    def retry_cleanup(self):
        calls.append("retry-cleanup")
        return _refused_result(self)

    monkeypatch.setattr(RSMOperationExecution, "run", held_run)
    monkeypatch.setattr(RSMOperationExecution, "retry_cleanup", retry_cleanup)
    owner = _prepare_owner(preflight)
    assert owner.begin_run() is not None
    assert entered.wait(2)
    progress = owner.poll()
    assert (progress.progress.revision, progress.progress.completed) == (2, 2)
    assert owner.poll() is None
    release.set()
    pending = _poll_terminal(owner)
    assert pending.outcome.kind is RSMOwnerOutcomeKind.CLEANUP_PENDING
    assert owner.finalization is RSMOwnerFinalization.CLEANUP_PENDING
    assert owner.begin_retry_cleanup() is not None
    completed = _poll_terminal(owner)
    assert completed.outcome.kind is RSMOwnerOutcomeKind.RESULT
    assert calls == ["run", "retry-cleanup"]
    assert owner.close().status is PageCleanup.CLEAN


def test_verification_retry_uses_retained_execution(
    monkeypatch,
    prepared_rsm_tool,
):
    preflight = prepared_rsm_tool
    calls = []

    def run(self, *, cancel_token=None, progress_callback=None):
        calls.append("run")
        raise RSMOperationVerificationError(self, "held reload")

    def retry_verification(self):
        calls.append("retry-verification")
        return _refused_result(self)

    monkeypatch.setattr(RSMOperationExecution, "run", run)
    monkeypatch.setattr(
        RSMOperationExecution,
        "retry_verification",
        retry_verification,
    )
    owner = _prepare_owner(preflight)
    assert owner.begin_run() is not None
    pending = _poll_terminal(owner)
    assert pending.outcome.kind is RSMOwnerOutcomeKind.VERIFICATION_PENDING
    assert owner.begin_retry_verification() is not None
    assert _poll_terminal(owner).outcome.kind is RSMOwnerOutcomeKind.RESULT
    assert calls == ["run", "retry-verification"]
    assert owner.close().status is PageCleanup.CLEAN


def test_close_drains_and_retries_exact_cleanup(monkeypatch, prepared_rsm_tool):
    preflight = prepared_rsm_tool
    entered = Event()
    release = Event()
    calls = []

    def held_run(self, *, cancel_token=None, progress_callback=None):
        calls.append(("run", self))
        entered.set()
        assert release.wait(2)
        assert cancel_token.is_set()
        raise RSMOperationCleanupPending(self)

    def retry_cleanup(self):
        calls.append(("retry-cleanup", self))
        return _refused_result(self)

    monkeypatch.setattr(RSMOperationExecution, "run", held_run)
    monkeypatch.setattr(RSMOperationExecution, "retry_cleanup", retry_cleanup)
    owner = _prepare_owner(preflight)
    assert owner.begin_run() is not None
    assert entered.wait(2)
    assert owner.close().status is PageCleanup.PENDING
    release.set()
    deadline = time.monotonic() + 3
    while True:
        receipt = owner.close()
        if receipt.status is PageCleanup.CLEAN:
            break
        assert time.monotonic() < deadline
        time.sleep(0.001)
    assert [name for name, _execution in calls] == ["run", "retry-cleanup"]
    assert calls[0][1] is calls[1][1]


def test_close_drains_and_retries_exact_verification(
    monkeypatch,
    prepared_rsm_tool,
):
    preflight = prepared_rsm_tool
    calls = []

    def run(self, *, cancel_token=None, progress_callback=None):
        calls.append(("run", self))
        raise RSMOperationVerificationError(self, "held reload")

    def retry_verification(self):
        calls.append(("retry-verification", self))
        return _refused_result(self)

    monkeypatch.setattr(RSMOperationExecution, "run", run)
    monkeypatch.setattr(
        RSMOperationExecution,
        "retry_verification",
        retry_verification,
    )
    owner = _prepare_owner(preflight)
    assert owner.begin_run() is not None
    pending = _poll_terminal(owner)
    assert pending.outcome.kind is RSMOwnerOutcomeKind.VERIFICATION_PENDING
    assert owner.close().status is PageCleanup.PENDING

    deadline = time.monotonic() + 3
    while True:
        receipt = owner.close()
        if receipt.status is PageCleanup.CLEAN:
            break
        assert time.monotonic() < deadline
        time.sleep(0.001)

    assert [name for name, _execution in calls] == [
        "run",
        "retry-verification",
    ]
    assert calls[0][1] is calls[1][1]
